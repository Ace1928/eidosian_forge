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
class HistoryManager(HistoryAccessor):
    """A class to organize all history-related functionality in one place.
    """
    shell = Instance('IPython.core.interactiveshell.InteractiveShellABC', allow_none=True)
    input_hist_parsed = List([''])
    input_hist_raw = List([''])
    dir_hist: List = List()

    @default('dir_hist')
    def _dir_hist_default(self):
        try:
            return [Path.cwd()]
        except OSError:
            return []
    output_hist = Dict()
    output_hist_reprs = Dict()
    session_number = Integer()
    db_log_output = Bool(False, help='Should the history database include output? (default: no)').tag(config=True)
    db_cache_size = Integer(0, help='Write to database every x commands (higher values save disk access & power).\nValues of 1 or less effectively disable caching.').tag(config=True)
    db_input_cache: List = List()
    db_output_cache: List = List()
    save_thread = Instance('IPython.core.history.HistorySavingThread', allow_none=True)
    save_flag = Instance(threading.Event, allow_none=True)
    _i00 = Unicode('')
    _i = Unicode('')
    _ii = Unicode('')
    _iii = Unicode('')
    _exit_re = re.compile('(exit|quit)(\\s*\\(.*\\))?$')

    def __init__(self, shell=None, config=None, **traits):
        """Create a new history manager associated with a shell instance.
        """
        super(HistoryManager, self).__init__(shell=shell, config=config, **traits)
        self.save_flag = threading.Event()
        self.db_input_cache_lock = threading.Lock()
        self.db_output_cache_lock = threading.Lock()
        try:
            self.new_session()
        except sqlite3.OperationalError:
            self.log.error('Failed to create history session in %s. History will not be saved.', self.hist_file, exc_info=True)
            self.hist_file = ':memory:'
        if self.enabled and self.hist_file != ':memory:':
            self.save_thread = HistorySavingThread(self)
            try:
                self.save_thread.start()
            except RuntimeError:
                self.log.error('Failed to start history saving thread. History will not be saved.', exc_info=True)
                self.hist_file = ':memory:'

    def _get_hist_file_name(self, profile=None):
        """Get default history file name based on the Shell's profile.

        The profile parameter is ignored, but must exist for compatibility with
        the parent class."""
        profile_dir = self.shell.profile_dir.location
        return Path(profile_dir) / 'history.sqlite'

    @only_when_enabled
    def new_session(self, conn=None):
        """Get a new session number."""
        if conn is None:
            conn = self.db
        with conn:
            cur = conn.execute("INSERT INTO sessions VALUES (NULL, ?, NULL,\n                            NULL, '') ", (datetime.datetime.now().isoformat(' '),))
            self.session_number = cur.lastrowid

    def end_session(self):
        """Close the database session, filling in the end time and line count."""
        self.writeout_cache()
        with self.db:
            self.db.execute('UPDATE sessions SET end=?, num_cmds=? WHERE\n                            session==?', (datetime.datetime.now().isoformat(' '), len(self.input_hist_parsed) - 1, self.session_number))
        self.session_number = 0

    def name_session(self, name):
        """Give the current session a name in the history database."""
        with self.db:
            self.db.execute('UPDATE sessions SET remark=? WHERE session==?', (name, self.session_number))

    def reset(self, new_session=True):
        """Clear the session history, releasing all object references, and
        optionally open a new session."""
        self.output_hist.clear()
        self.dir_hist[:] = [Path.cwd()]
        if new_session:
            if self.session_number:
                self.end_session()
            self.input_hist_parsed[:] = ['']
            self.input_hist_raw[:] = ['']
            self.new_session()

    def get_session_info(self, session=0):
        """Get info about a session.

        Parameters
        ----------
        session : int
            Session number to retrieve. The current session is 0, and negative
            numbers count back from current session, so -1 is the previous session.

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
        if session <= 0:
            session += self.session_number
        return super(HistoryManager, self).get_session_info(session=session)

    @catch_corrupt_db
    def get_tail(self, n=10, raw=True, output=False, include_latest=False):
        """Get the last n lines from the history database.

        Most recent entry last.

        Completion will be reordered so that that the last ones are when
        possible from current session.

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
        this_cur = list(self._run_sql('WHERE session == ? ORDER BY line DESC LIMIT ?  ', (self.session_number, n), raw=raw, output=output))
        other_cur = list(self._run_sql('WHERE session != ? ORDER BY session DESC, line DESC LIMIT ?', (self.session_number, n), raw=raw, output=output))
        everything = this_cur + other_cur
        everything = everything[:n]
        if not include_latest:
            return list(everything)[:0:-1]
        return list(everything)[::-1]

    def _get_range_session(self, start=1, stop=None, raw=True, output=False):
        """Get input and output history from the current session. Called by
        get_range, and takes similar parameters."""
        input_hist = self.input_hist_raw if raw else self.input_hist_parsed
        n = len(input_hist)
        if start < 0:
            start += n
        if not stop or stop > n:
            stop = n
        elif stop < 0:
            stop += n
        for i in range(start, stop):
            if output:
                line = (input_hist[i], self.output_hist_reprs.get(i))
            else:
                line = input_hist[i]
            yield (0, i, line)

    def get_range(self, session=0, start=1, stop=None, raw=True, output=False):
        """Retrieve input by session.

        Parameters
        ----------
        session : int
            Session number to retrieve. The current session is 0, and negative
            numbers count back from current session, so -1 is previous session.
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
        if session <= 0:
            session += self.session_number
        if session == self.session_number:
            return self._get_range_session(start, stop, raw, output)
        return super(HistoryManager, self).get_range(session, start, stop, raw, output)

    def store_inputs(self, line_num, source, source_raw=None):
        """Store source and raw input in history and create input cache
        variables ``_i*``.

        Parameters
        ----------
        line_num : int
            The prompt number of this input.
        source : str
            Python input.
        source_raw : str, optional
            If given, this is the raw input without any IPython transformations
            applied to it.  If not given, ``source`` is used.
        """
        if source_raw is None:
            source_raw = source
        source = source.rstrip('\n')
        source_raw = source_raw.rstrip('\n')
        if self._exit_re.match(source_raw.strip()):
            return
        self.input_hist_parsed.append(source)
        self.input_hist_raw.append(source_raw)
        with self.db_input_cache_lock:
            self.db_input_cache.append((line_num, source, source_raw))
            if len(self.db_input_cache) >= self.db_cache_size:
                self.save_flag.set()
        self._iii = self._ii
        self._ii = self._i
        self._i = self._i00
        self._i00 = source_raw
        new_i = '_i%s' % line_num
        to_main = {'_i': self._i, '_ii': self._ii, '_iii': self._iii, new_i: self._i00}
        if self.shell is not None:
            self.shell.push(to_main, interactive=False)

    def store_output(self, line_num):
        """If database output logging is enabled, this saves all the
        outputs from the indicated prompt number to the database. It's
        called by run_cell after code has been executed.

        Parameters
        ----------
        line_num : int
            The line number from which to save outputs
        """
        if not self.db_log_output or line_num not in self.output_hist_reprs:
            return
        output = self.output_hist_reprs[line_num]
        with self.db_output_cache_lock:
            self.db_output_cache.append((line_num, output))
        if self.db_cache_size <= 1:
            self.save_flag.set()

    def _writeout_input_cache(self, conn):
        with conn:
            for line in self.db_input_cache:
                conn.execute('INSERT INTO history VALUES (?, ?, ?, ?)', (self.session_number,) + line)

    def _writeout_output_cache(self, conn):
        with conn:
            for line in self.db_output_cache:
                conn.execute('INSERT INTO output_history VALUES (?, ?, ?)', (self.session_number,) + line)

    @only_when_enabled
    def writeout_cache(self, conn=None):
        """Write any entries in the cache to the database."""
        if conn is None:
            conn = self.db
        with self.db_input_cache_lock:
            try:
                self._writeout_input_cache(conn)
            except sqlite3.IntegrityError:
                self.new_session(conn)
                print('ERROR! Session/line number was not unique in', 'database. History logging moved to new session', self.session_number)
                try:
                    self._writeout_input_cache(conn)
                except sqlite3.IntegrityError:
                    pass
            finally:
                self.db_input_cache = []
        with self.db_output_cache_lock:
            try:
                self._writeout_output_cache(conn)
            except sqlite3.IntegrityError:
                print('!! Session/line number for output was not unique', 'in database. Output will not be stored.')
            finally:
                self.db_output_cache = []