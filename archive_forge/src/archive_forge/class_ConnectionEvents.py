from __future__ import annotations
import typing
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union
from .base import Connection
from .base import Engine
from .interfaces import ConnectionEventsTarget
from .interfaces import DBAPIConnection
from .interfaces import DBAPICursor
from .interfaces import Dialect
from .. import event
from .. import exc
from ..util.typing import Literal
class ConnectionEvents(event.Events[ConnectionEventsTarget]):
    """Available events for
    :class:`_engine.Connection` and :class:`_engine.Engine`.

    The methods here define the name of an event as well as the names of
    members that are passed to listener functions.

    An event listener can be associated with any
    :class:`_engine.Connection` or :class:`_engine.Engine`
    class or instance, such as an :class:`_engine.Engine`, e.g.::

        from sqlalchemy import event, create_engine

        def before_cursor_execute(conn, cursor, statement, parameters, context,
                                                        executemany):
            log.info("Received statement: %s", statement)

        engine = create_engine('postgresql+psycopg2://scott:tiger@localhost/test')
        event.listen(engine, "before_cursor_execute", before_cursor_execute)

    or with a specific :class:`_engine.Connection`::

        with engine.begin() as conn:
            @event.listens_for(conn, 'before_cursor_execute')
            def before_cursor_execute(conn, cursor, statement, parameters,
                                            context, executemany):
                log.info("Received statement: %s", statement)

    When the methods are called with a `statement` parameter, such as in
    :meth:`.after_cursor_execute` or :meth:`.before_cursor_execute`,
    the statement is the exact SQL string that was prepared for transmission
    to the DBAPI ``cursor`` in the connection's :class:`.Dialect`.

    The :meth:`.before_execute` and :meth:`.before_cursor_execute`
    events can also be established with the ``retval=True`` flag, which
    allows modification of the statement and parameters to be sent
    to the database.  The :meth:`.before_cursor_execute` event is
    particularly useful here to add ad-hoc string transformations, such
    as comments, to all executions::

        from sqlalchemy.engine import Engine
        from sqlalchemy import event

        @event.listens_for(Engine, "before_cursor_execute", retval=True)
        def comment_sql_calls(conn, cursor, statement, parameters,
                                            context, executemany):
            statement = statement + " -- some comment"
            return statement, parameters

    .. note:: :class:`_events.ConnectionEvents` can be established on any
       combination of :class:`_engine.Engine`, :class:`_engine.Connection`,
       as well
       as instances of each of those classes.  Events across all
       four scopes will fire off for a given instance of
       :class:`_engine.Connection`.  However, for performance reasons, the
       :class:`_engine.Connection` object determines at instantiation time
       whether or not its parent :class:`_engine.Engine` has event listeners
       established.   Event listeners added to the :class:`_engine.Engine`
       class or to an instance of :class:`_engine.Engine`
       *after* the instantiation
       of a dependent :class:`_engine.Connection` instance will usually
       *not* be available on that :class:`_engine.Connection` instance.
       The newly
       added listeners will instead take effect for
       :class:`_engine.Connection`
       instances created subsequent to those event listeners being
       established on the parent :class:`_engine.Engine` class or instance.

    :param retval=False: Applies to the :meth:`.before_execute` and
      :meth:`.before_cursor_execute` events only.  When True, the
      user-defined event function must have a return value, which
      is a tuple of parameters that replace the given statement
      and parameters.  See those methods for a description of
      specific return arguments.

    """
    _target_class_doc = 'SomeEngine'
    _dispatch_target = ConnectionEventsTarget

    @classmethod
    def _accept_with(cls, target: Union[ConnectionEventsTarget, Type[ConnectionEventsTarget]], identifier: str) -> Optional[Union[ConnectionEventsTarget, Type[ConnectionEventsTarget]]]:
        default_dispatch = super()._accept_with(target, identifier)
        if default_dispatch is None and hasattr(target, '_no_async_engine_events'):
            target._no_async_engine_events()
        return default_dispatch

    @classmethod
    def _listen(cls, event_key: event._EventKey[ConnectionEventsTarget], *, retval: bool=False, **kw: Any) -> None:
        target, identifier, fn = (event_key.dispatch_target, event_key.identifier, event_key._listen_fn)
        target._has_events = True
        if not retval:
            if identifier == 'before_execute':
                orig_fn = fn

                def wrap_before_execute(conn, clauseelement, multiparams, params, execution_options):
                    orig_fn(conn, clauseelement, multiparams, params, execution_options)
                    return (clauseelement, multiparams, params)
                fn = wrap_before_execute
            elif identifier == 'before_cursor_execute':
                orig_fn = fn

                def wrap_before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
                    orig_fn(conn, cursor, statement, parameters, context, executemany)
                    return (statement, parameters)
                fn = wrap_before_cursor_execute
        elif retval and identifier not in ('before_execute', 'before_cursor_execute'):
            raise exc.ArgumentError("Only the 'before_execute', 'before_cursor_execute' and 'handle_error' engine event listeners accept the 'retval=True' argument.")
        event_key.with_wrapper(fn).base_listen()

    @event._legacy_signature('1.4', ['conn', 'clauseelement', 'multiparams', 'params'], lambda conn, clauseelement, multiparams, params, execution_options: (conn, clauseelement, multiparams, params))
    def before_execute(self, conn: Connection, clauseelement: Executable, multiparams: _CoreMultiExecuteParams, params: _CoreSingleExecuteParams, execution_options: _ExecuteOptions) -> Optional[Tuple[Executable, _CoreMultiExecuteParams, _CoreSingleExecuteParams]]:
        """Intercept high level execute() events, receiving uncompiled
        SQL constructs and other objects prior to rendering into SQL.

        This event is good for debugging SQL compilation issues as well
        as early manipulation of the parameters being sent to the database,
        as the parameter lists will be in a consistent format here.

        This event can be optionally established with the ``retval=True``
        flag.  The ``clauseelement``, ``multiparams``, and ``params``
        arguments should be returned as a three-tuple in this case::

            @event.listens_for(Engine, "before_execute", retval=True)
            def before_execute(conn, clauseelement, multiparams, params):
                # do something with clauseelement, multiparams, params
                return clauseelement, multiparams, params

        :param conn: :class:`_engine.Connection` object
        :param clauseelement: SQL expression construct, :class:`.Compiled`
         instance, or string statement passed to
         :meth:`_engine.Connection.execute`.
        :param multiparams: Multiple parameter sets, a list of dictionaries.
        :param params: Single parameter set, a single dictionary.
        :param execution_options: dictionary of execution
         options passed along with the statement, if any.  This is a merge
         of all options that will be used, including those of the statement,
         the connection, and those passed in to the method itself for
         the 2.0 style of execution.

         .. versionadded: 1.4

        .. seealso::

            :meth:`.before_cursor_execute`

        """

    @event._legacy_signature('1.4', ['conn', 'clauseelement', 'multiparams', 'params', 'result'], lambda conn, clauseelement, multiparams, params, execution_options, result: (conn, clauseelement, multiparams, params, result))
    def after_execute(self, conn: Connection, clauseelement: Executable, multiparams: _CoreMultiExecuteParams, params: _CoreSingleExecuteParams, execution_options: _ExecuteOptions, result: Result[Any]) -> None:
        """Intercept high level execute() events after execute.


        :param conn: :class:`_engine.Connection` object
        :param clauseelement: SQL expression construct, :class:`.Compiled`
         instance, or string statement passed to
         :meth:`_engine.Connection.execute`.
        :param multiparams: Multiple parameter sets, a list of dictionaries.
        :param params: Single parameter set, a single dictionary.
        :param execution_options: dictionary of execution
         options passed along with the statement, if any.  This is a merge
         of all options that will be used, including those of the statement,
         the connection, and those passed in to the method itself for
         the 2.0 style of execution.

         .. versionadded: 1.4

        :param result: :class:`_engine.CursorResult` generated by the
         execution.

        """

    def before_cursor_execute(self, conn: Connection, cursor: DBAPICursor, statement: str, parameters: _DBAPIAnyExecuteParams, context: Optional[ExecutionContext], executemany: bool) -> Optional[Tuple[str, _DBAPIAnyExecuteParams]]:
        """Intercept low-level cursor execute() events before execution,
        receiving the string SQL statement and DBAPI-specific parameter list to
        be invoked against a cursor.

        This event is a good choice for logging as well as late modifications
        to the SQL string.  It's less ideal for parameter modifications except
        for those which are specific to a target backend.

        This event can be optionally established with the ``retval=True``
        flag.  The ``statement`` and ``parameters`` arguments should be
        returned as a two-tuple in this case::

            @event.listens_for(Engine, "before_cursor_execute", retval=True)
            def before_cursor_execute(conn, cursor, statement,
                            parameters, context, executemany):
                # do something with statement, parameters
                return statement, parameters

        See the example at :class:`_events.ConnectionEvents`.

        :param conn: :class:`_engine.Connection` object
        :param cursor: DBAPI cursor object
        :param statement: string SQL statement, as to be passed to the DBAPI
        :param parameters: Dictionary, tuple, or list of parameters being
         passed to the ``execute()`` or ``executemany()`` method of the
         DBAPI ``cursor``.  In some cases may be ``None``.
        :param context: :class:`.ExecutionContext` object in use.  May
         be ``None``.
        :param executemany: boolean, if ``True``, this is an ``executemany()``
         call, if ``False``, this is an ``execute()`` call.

        .. seealso::

            :meth:`.before_execute`

            :meth:`.after_cursor_execute`

        """

    def after_cursor_execute(self, conn: Connection, cursor: DBAPICursor, statement: str, parameters: _DBAPIAnyExecuteParams, context: Optional[ExecutionContext], executemany: bool) -> None:
        """Intercept low-level cursor execute() events after execution.

        :param conn: :class:`_engine.Connection` object
        :param cursor: DBAPI cursor object.  Will have results pending
         if the statement was a SELECT, but these should not be consumed
         as they will be needed by the :class:`_engine.CursorResult`.
        :param statement: string SQL statement, as passed to the DBAPI
        :param parameters: Dictionary, tuple, or list of parameters being
         passed to the ``execute()`` or ``executemany()`` method of the
         DBAPI ``cursor``.  In some cases may be ``None``.
        :param context: :class:`.ExecutionContext` object in use.  May
         be ``None``.
        :param executemany: boolean, if ``True``, this is an ``executemany()``
         call, if ``False``, this is an ``execute()`` call.

        """

    @event._legacy_signature('2.0', ['conn', 'branch'], converter=lambda conn: (conn, False))
    def engine_connect(self, conn: Connection) -> None:
        """Intercept the creation of a new :class:`_engine.Connection`.

        This event is called typically as the direct result of calling
        the :meth:`_engine.Engine.connect` method.

        It differs from the :meth:`_events.PoolEvents.connect` method, which
        refers to the actual connection to a database at the DBAPI level;
        a DBAPI connection may be pooled and reused for many operations.
        In contrast, this event refers only to the production of a higher level
        :class:`_engine.Connection` wrapper around such a DBAPI connection.

        It also differs from the :meth:`_events.PoolEvents.checkout` event
        in that it is specific to the :class:`_engine.Connection` object,
        not the
        DBAPI connection that :meth:`_events.PoolEvents.checkout` deals with,
        although
        this DBAPI connection is available here via the
        :attr:`_engine.Connection.connection` attribute.
        But note there can in fact
        be multiple :meth:`_events.PoolEvents.checkout`
        events within the lifespan
        of a single :class:`_engine.Connection` object, if that
        :class:`_engine.Connection`
        is invalidated and re-established.

        :param conn: :class:`_engine.Connection` object.

        .. seealso::

            :meth:`_events.PoolEvents.checkout`
            the lower-level pool checkout event
            for an individual DBAPI connection

        """

    def set_connection_execution_options(self, conn: Connection, opts: Dict[str, Any]) -> None:
        """Intercept when the :meth:`_engine.Connection.execution_options`
        method is called.

        This method is called after the new :class:`_engine.Connection`
        has been
        produced, with the newly updated execution options collection, but
        before the :class:`.Dialect` has acted upon any of those new options.

        Note that this method is not called when a new
        :class:`_engine.Connection`
        is produced which is inheriting execution options from its parent
        :class:`_engine.Engine`; to intercept this condition, use the
        :meth:`_events.ConnectionEvents.engine_connect` event.

        :param conn: The newly copied :class:`_engine.Connection` object

        :param opts: dictionary of options that were passed to the
         :meth:`_engine.Connection.execution_options` method.
         This dictionary may be modified in place to affect the ultimate
         options which take effect.

         .. versionadded:: 2.0 the ``opts`` dictionary may be modified
            in place.


        .. seealso::

            :meth:`_events.ConnectionEvents.set_engine_execution_options`
            - event
            which is called when :meth:`_engine.Engine.execution_options`
            is called.


        """

    def set_engine_execution_options(self, engine: Engine, opts: Dict[str, Any]) -> None:
        """Intercept when the :meth:`_engine.Engine.execution_options`
        method is called.

        The :meth:`_engine.Engine.execution_options` method produces a shallow
        copy of the :class:`_engine.Engine` which stores the new options.
        That new
        :class:`_engine.Engine` is passed here.
        A particular application of this
        method is to add a :meth:`_events.ConnectionEvents.engine_connect`
        event
        handler to the given :class:`_engine.Engine`
        which will perform some per-
        :class:`_engine.Connection` task specific to these execution options.

        :param conn: The newly copied :class:`_engine.Engine` object

        :param opts: dictionary of options that were passed to the
         :meth:`_engine.Connection.execution_options` method.
         This dictionary may be modified in place to affect the ultimate
         options which take effect.

         .. versionadded:: 2.0 the ``opts`` dictionary may be modified
            in place.

        .. seealso::

            :meth:`_events.ConnectionEvents.set_connection_execution_options`
            - event
            which is called when :meth:`_engine.Connection.execution_options`
            is
            called.

        """

    def engine_disposed(self, engine: Engine) -> None:
        """Intercept when the :meth:`_engine.Engine.dispose` method is called.

        The :meth:`_engine.Engine.dispose` method instructs the engine to
        "dispose" of it's connection pool (e.g. :class:`_pool.Pool`), and
        replaces it with a new one.  Disposing of the old pool has the
        effect that existing checked-in connections are closed.  The new
        pool does not establish any new connections until it is first used.

        This event can be used to indicate that resources related to the
        :class:`_engine.Engine` should also be cleaned up,
        keeping in mind that the
        :class:`_engine.Engine`
        can still be used for new requests in which case
        it re-acquires connection resources.

        """

    def begin(self, conn: Connection) -> None:
        """Intercept begin() events.

        :param conn: :class:`_engine.Connection` object

        """

    def rollback(self, conn: Connection) -> None:
        """Intercept rollback() events, as initiated by a
        :class:`.Transaction`.

        Note that the :class:`_pool.Pool` also "auto-rolls back"
        a DBAPI connection upon checkin, if the ``reset_on_return``
        flag is set to its default value of ``'rollback'``.
        To intercept this
        rollback, use the :meth:`_events.PoolEvents.reset` hook.

        :param conn: :class:`_engine.Connection` object

        .. seealso::

            :meth:`_events.PoolEvents.reset`

        """

    def commit(self, conn: Connection) -> None:
        """Intercept commit() events, as initiated by a
        :class:`.Transaction`.

        Note that the :class:`_pool.Pool` may also "auto-commit"
        a DBAPI connection upon checkin, if the ``reset_on_return``
        flag is set to the value ``'commit'``.  To intercept this
        commit, use the :meth:`_events.PoolEvents.reset` hook.

        :param conn: :class:`_engine.Connection` object
        """

    def savepoint(self, conn: Connection, name: str) -> None:
        """Intercept savepoint() events.

        :param conn: :class:`_engine.Connection` object
        :param name: specified name used for the savepoint.

        """

    def rollback_savepoint(self, conn: Connection, name: str, context: None) -> None:
        """Intercept rollback_savepoint() events.

        :param conn: :class:`_engine.Connection` object
        :param name: specified name used for the savepoint.
        :param context: not used

        """

    def release_savepoint(self, conn: Connection, name: str, context: None) -> None:
        """Intercept release_savepoint() events.

        :param conn: :class:`_engine.Connection` object
        :param name: specified name used for the savepoint.
        :param context: not used

        """

    def begin_twophase(self, conn: Connection, xid: Any) -> None:
        """Intercept begin_twophase() events.

        :param conn: :class:`_engine.Connection` object
        :param xid: two-phase XID identifier

        """

    def prepare_twophase(self, conn: Connection, xid: Any) -> None:
        """Intercept prepare_twophase() events.

        :param conn: :class:`_engine.Connection` object
        :param xid: two-phase XID identifier
        """

    def rollback_twophase(self, conn: Connection, xid: Any, is_prepared: bool) -> None:
        """Intercept rollback_twophase() events.

        :param conn: :class:`_engine.Connection` object
        :param xid: two-phase XID identifier
        :param is_prepared: boolean, indicates if
         :meth:`.TwoPhaseTransaction.prepare` was called.

        """

    def commit_twophase(self, conn: Connection, xid: Any, is_prepared: bool) -> None:
        """Intercept commit_twophase() events.

        :param conn: :class:`_engine.Connection` object
        :param xid: two-phase XID identifier
        :param is_prepared: boolean, indicates if
         :meth:`.TwoPhaseTransaction.prepare` was called.

        """