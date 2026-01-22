import atexit
import datetime
import re
import sqlite3
import threading
from pathlib import Path
from decorator import decorator
from traitlets import (
from traitlets.config.configurable import LoggingConfigurable
from IPython.paths import locate_profile
from IPython.utils.decorators import undoc
class HistoryAccessor(HistoryAccessorBase):
    """Access the history database without adding to it.

    This is intended for use by standalone history tools. IPython shells use
    HistoryManager, below, which is a subclass of this."""
    _corrupt_db_counter = 0
    _corrupt_db_limit = 2
    hist_file = Union([Instance(Path), Unicode()], help='Path to file to use for SQLite history database.\n\n        By default, IPython will put the history database in the IPython\n        profile directory.  If you would rather share one history among\n        profiles, you can set this value in each, so that they are consistent.\n\n        Due to an issue with fcntl, SQLite is known to misbehave on some NFS\n        mounts.  If you see IPython hanging, try setting this to something on a\n        local disk, e.g::\n\n            ipython --HistoryManager.hist_file=/tmp/ipython_hist.sqlite\n\n        you can also use the specific value `:memory:` (including the colon\n        at both end but not the back ticks), to avoid creating an history file.\n\n        ').tag(config=True)
    enabled = Bool(True, help='enable the SQLite history\n\n        set enabled=False to disable the SQLite history,\n        in which case there will be no stored history, no SQLite connection,\n        and no background saving thread.  This may be necessary in some\n        threaded environments where IPython is embedded.\n        ').tag(config=True)
    connection_options = Dict(help='Options for configuring the SQLite connection\n\n        These options are passed as keyword args to sqlite3.connect\n        when establishing database connections.\n        ').tag(config=True)

    @default('connection_options')
    def _default_connection_options(self):
        return dict(check_same_thread=False)
    db = Any()

    @observe('db')
    def _db_changed(self, change):
        """validate the db, since it can be an Instance of two different types"""
        new = change['new']
        connection_types = (DummyDB, sqlite3.Connection)
        if not isinstance(new, connection_types):
            msg = '%s.db must be sqlite3 Connection or DummyDB, not %r' % (self.__class__.__name__, new)
            raise TraitError(msg)

    def __init__(self, profile='default', hist_file='', **traits):
        """Create a new history accessor.

        Parameters
        ----------
        profile : str
            The name of the profile from which to open history.
        hist_file : str
            Path to an SQLite history database stored by IPython. If specified,
            hist_file overrides profile.
        config : :class:`~traitlets.config.loader.Config`
            Config object. hist_file can also be set through this.
        """
        super(HistoryAccessor, self).__init__(**traits)
        if hist_file:
            self.hist_file = hist_file
        try:
            self.hist_file
        except TraitError:
            self.hist_file = self._get_hist_file_name(profile)
        self.init_db()

    def _get_hist_file_name(self, profile='default'):
        """Find the history file for the given profile name.

        This is overridden by the HistoryManager subclass, to use the shell's
        active profile.

        Parameters
        ----------
        profile : str
            The name of a profile which has a history file.
        """
        return Path(locate_profile(profile)) / 'history.sqlite'

    @catch_corrupt_db
    def init_db(self):
        """Connect to the database, and create tables if necessary."""
        if not self.enabled:
            self.db = DummyDB()
            return
        kwargs = dict(detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
        kwargs.update(self.connection_options)
        self.db = sqlite3.connect(str(self.hist_file), **kwargs)
        with self.db:
            self.db.execute('CREATE TABLE IF NOT EXISTS sessions (session integer\n                            primary key autoincrement, start timestamp,\n                            end timestamp, num_cmds integer, remark text)')
            self.db.execute('CREATE TABLE IF NOT EXISTS history\n                    (session integer, line integer, source text, source_raw text,\n                    PRIMARY KEY (session, line))')
            self.db.execute('CREATE TABLE IF NOT EXISTS output_history\n                            (session integer, line integer, output text,\n                            PRIMARY KEY (session, line))')
        self._corrupt_db_counter = 0

    def writeout_cache(self):
        """Overridden by HistoryManager to dump the cache before certain
        database lookups."""
        pass

    def _run_sql(self, sql, params, raw=True, output=False, latest=False):
        """Prepares and runs an SQL query for the history database.

        Parameters
        ----------
        sql : str
            Any filtering expressions to go after SELECT ... FROM ...
        params : tuple
            Parameters passed to the SQL query (to replace "?")
        raw, output : bool
            See :meth:`get_range`
        latest : bool
            Select rows with max (session, line)

        Returns
        -------
        Tuples as :meth:`get_range`
        """
        toget = 'source_raw' if raw else 'source'
        sqlfrom = 'history'
        if output:
            sqlfrom = 'history LEFT JOIN output_history USING (session, line)'
            toget = 'history.%s, output_history.output' % toget
        if latest:
            toget += ', MAX(session * 128 * 1024 + line)'
        this_querry = 'SELECT session, line, %s FROM %s ' % (toget, sqlfrom) + sql
        cur = self.db.execute(this_querry, params)
        if latest:
            cur = (row[:-1] for row in cur)
        if output:
            return ((ses, lin, (inp, out)) for ses, lin, inp, out in cur)
        return cur

    @only_when_enabled
    @catch_corrupt_db
    def get_session_info(self, session):
        """Get info about a session.

        Parameters
        ----------
        session : int
            Session number to retrieve.

        Returns
        -------
        session_id : int
            Session ID number
        start : datetime
            Timestamp for the start of the session.
        end : datetime
            Timestamp for the end of the session, or None if IPython crashed.
        num_cmds : int
            Number of commands run, or None if IPython crashed.
        remark : unicode
            A manually set description.
        """
        query = 'SELECT * from sessions where session == ?'
        return self.db.execute(query, (session,)).fetchone()

    @catch_corrupt_db
    def get_last_session_id(self):
        """Get the last session ID currently in the database.

        Within IPython, this should be the same as the value stored in
        :attr:`HistoryManager.session_number`.
        """
        for record in self.get_tail(n=1, include_latest=True):
            return record[0]

    @catch_corrupt_db
    def get_tail(self, n=10, raw=True, output=False, include_latest=False):
        """Get the last n lines from the history database.

        Parameters
        ----------
        n : int
            The number of lines to get
        raw, output : bool
            See :meth:`get_range`
        include_latest : bool
            If False (default), n+1 lines are fetched, and the latest one
            is discarded. This is intended to be used where the function
            is called by a user command, which it should not return.

        Returns
        -------
        Tuples as :meth:`get_range`
        """
        self.writeout_cache()
        if not include_latest:
            n += 1
        cur = self._run_sql('ORDER BY session DESC, line DESC LIMIT ?', (n,), raw=raw, output=output)
        if not include_latest:
            return reversed(list(cur)[1:])
        return reversed(list(cur))

    @catch_corrupt_db
    def search(self, pattern='*', raw=True, search_raw=True, output=False, n=None, unique=False):
        """Search the database using unix glob-style matching (wildcards
        * and ?).

        Parameters
        ----------
        pattern : str
            The wildcarded pattern to match when searching
        search_raw : bool
            If True, search the raw input, otherwise, the parsed input
        raw, output : bool
            See :meth:`get_range`
        n : None or int
            If an integer is given, it defines the limit of
            returned entries.
        unique : bool
            When it is true, return only unique entries.

        Returns
        -------
        Tuples as :meth:`get_range`
        """
        tosearch = 'source_raw' if search_raw else 'source'
        if output:
            tosearch = 'history.' + tosearch
        self.writeout_cache()
        sqlform = 'WHERE %s GLOB ?' % tosearch
        params = (pattern,)
        if unique:
            sqlform += ' GROUP BY {0}'.format(tosearch)
        if n is not None:
            sqlform += ' ORDER BY session DESC, line DESC LIMIT ?'
            params += (n,)
        elif unique:
            sqlform += ' ORDER BY session, line'
        cur = self._run_sql(sqlform, params, raw=raw, output=output, latest=unique)
        if n is not None:
            return reversed(list(cur))
        return cur

    @catch_corrupt_db
    def get_range(self, session, start=1, stop=None, raw=True, output=False):
        """Retrieve input by session.

        Parameters
        ----------
        session : int
            Session number to retrieve.
        start : int
            First line to retrieve.
        stop : int
            End of line range (excluded from output itself). If None, retrieve
            to the end of the session.
        raw : bool
            If True, return untranslated input
        output : bool
            If True, attempt to include output. This will be 'real' Python
            objects for the current session, or text reprs from previous
            sessions if db_log_output was enabled at the time. Where no output
            is found, None is used.

        Returns
        -------
        entries
            An iterator over the desired lines. Each line is a 3-tuple, either
            (session, line, input) if output is False, or
            (session, line, (input, output)) if output is True.
        """
        if stop:
            lineclause = 'line >= ? AND line < ?'
            params = (session, start, stop)
        else:
            lineclause = 'line>=?'
            params = (session, start)
        return self._run_sql('WHERE session==? AND %s' % lineclause, params, raw=raw, output=output)

    def get_range_by_str(self, rangestr, raw=True, output=False):
        """Get lines of history from a string of ranges, as used by magic
        commands %hist, %save, %macro, etc.

        Parameters
        ----------
        rangestr : str
            A string specifying ranges, e.g. "5 ~2/1-4". If empty string is used,
            this will return everything from current session's history.

            See the documentation of :func:`%history` for the full details.

        raw, output : bool
            As :meth:`get_range`

        Returns
        -------
        Tuples as :meth:`get_range`
        """
        for sess, s, e in extract_hist_ranges(rangestr):
            for line in self.get_range(sess, s, e, raw=raw, output=output):
                yield line