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
class DialectEvents(event.Events[Dialect]):
    """event interface for execution-replacement functions.

    These events allow direct instrumentation and replacement
    of key dialect functions which interact with the DBAPI.

    .. note::

        :class:`.DialectEvents` hooks should be considered **semi-public**
        and experimental.
        These hooks are not for general use and are only for those situations
        where intricate re-statement of DBAPI mechanics must be injected onto
        an existing dialect.  For general-use statement-interception events,
        please use the :class:`_events.ConnectionEvents` interface.

    .. seealso::

        :meth:`_events.ConnectionEvents.before_cursor_execute`

        :meth:`_events.ConnectionEvents.before_execute`

        :meth:`_events.ConnectionEvents.after_cursor_execute`

        :meth:`_events.ConnectionEvents.after_execute`

    """
    _target_class_doc = 'SomeEngine'
    _dispatch_target = Dialect

    @classmethod
    def _listen(cls, event_key: event._EventKey[Dialect], *, retval: bool=False, **kw: Any) -> None:
        target = event_key.dispatch_target
        target._has_events = True
        event_key.base_listen()

    @classmethod
    def _accept_with(cls, target: Union[Engine, Type[Engine], Dialect, Type[Dialect]], identifier: str) -> Optional[Union[Dialect, Type[Dialect]]]:
        if isinstance(target, type):
            if issubclass(target, Engine):
                return Dialect
            elif issubclass(target, Dialect):
                return target
        elif isinstance(target, Engine):
            return target.dialect
        elif isinstance(target, Dialect):
            return target
        elif isinstance(target, Connection) and identifier == 'handle_error':
            raise exc.InvalidRequestError('The handle_error() event hook as of SQLAlchemy 2.0 is established on the Dialect, and may only be applied to the Engine as a whole or to a specific Dialect as a whole, not on a per-Connection basis.')
        elif hasattr(target, '_no_async_engine_events'):
            target._no_async_engine_events()
        else:
            return None

    def handle_error(self, exception_context: ExceptionContext) -> Optional[BaseException]:
        """Intercept all exceptions processed by the
        :class:`_engine.Dialect`, typically but not limited to those
        emitted within the scope of a :class:`_engine.Connection`.

        .. versionchanged:: 2.0 the :meth:`.DialectEvents.handle_error` event
           is moved to the :class:`.DialectEvents` class, moved from the
           :class:`.ConnectionEvents` class, so that it may also participate in
           the "pre ping" operation configured with the
           :paramref:`_sa.create_engine.pool_pre_ping` parameter. The event
           remains registered by using the :class:`_engine.Engine` as the event
           target, however note that using the :class:`_engine.Connection` as
           an event target for :meth:`.DialectEvents.handle_error` is no longer
           supported.

        This includes all exceptions emitted by the DBAPI as well as
        within SQLAlchemy's statement invocation process, including
        encoding errors and other statement validation errors.  Other areas
        in which the event is invoked include transaction begin and end,
        result row fetching, cursor creation.

        Note that :meth:`.handle_error` may support new kinds of exceptions
        and new calling scenarios at *any time*.  Code which uses this
        event must expect new calling patterns to be present in minor
        releases.

        To support the wide variety of members that correspond to an exception,
        as well as to allow extensibility of the event without backwards
        incompatibility, the sole argument received is an instance of
        :class:`.ExceptionContext`.   This object contains data members
        representing detail about the exception.

        Use cases supported by this hook include:

        * read-only, low-level exception handling for logging and
          debugging purposes
        * Establishing whether a DBAPI connection error message indicates
          that the database connection needs to be reconnected, including
          for the "pre_ping" handler used by **some** dialects
        * Establishing or disabling whether a connection or the owning
          connection pool is invalidated or expired in response to a
          specific exception
        * exception re-writing

        The hook is called while the cursor from the failed operation
        (if any) is still open and accessible.   Special cleanup operations
        can be called on this cursor; SQLAlchemy will attempt to close
        this cursor subsequent to this hook being invoked.

        As of SQLAlchemy 2.0, the "pre_ping" handler enabled using the
        :paramref:`_sa.create_engine.pool_pre_ping` parameter will also
        participate in the :meth:`.handle_error` process, **for those dialects
        that rely upon disconnect codes to detect database liveness**. Note
        that some dialects such as psycopg, psycopg2, and most MySQL dialects
        make use of a native ``ping()`` method supplied by the DBAPI which does
        not make use of disconnect codes.

        .. versionchanged:: 2.0.0 The :meth:`.DialectEvents.handle_error`
           event hook participates in connection pool "pre-ping" operations.
           Within this usage, the :attr:`.ExceptionContext.engine` attribute
           will be ``None``, however the :class:`.Dialect` in use is always
           available via the :attr:`.ExceptionContext.dialect` attribute.

        .. versionchanged:: 2.0.5 Added :attr:`.ExceptionContext.is_pre_ping`
           attribute which will be set to ``True`` when the
           :meth:`.DialectEvents.handle_error` event hook is triggered within
           a connection pool pre-ping operation.

        .. versionchanged:: 2.0.5 An issue was repaired that allows for the
           PostgreSQL ``psycopg`` and ``psycopg2`` drivers, as well as all
           MySQL drivers, to properly participate in the
           :meth:`.DialectEvents.handle_error` event hook during
           connection pool "pre-ping" operations; previously, the
           implementation was non-working for these drivers.


        A handler function has two options for replacing
        the SQLAlchemy-constructed exception into one that is user
        defined.   It can either raise this new exception directly, in
        which case all further event listeners are bypassed and the
        exception will be raised, after appropriate cleanup as taken
        place::

            @event.listens_for(Engine, "handle_error")
            def handle_exception(context):
                if isinstance(context.original_exception,
                    psycopg2.OperationalError) and \\
                    "failed" in str(context.original_exception):
                    raise MySpecialException("failed operation")

        .. warning::  Because the
           :meth:`_events.DialectEvents.handle_error`
           event specifically provides for exceptions to be re-thrown as
           the ultimate exception raised by the failed statement,
           **stack traces will be misleading** if the user-defined event
           handler itself fails and throws an unexpected exception;
           the stack trace may not illustrate the actual code line that
           failed!  It is advised to code carefully here and use
           logging and/or inline debugging if unexpected exceptions are
           occurring.

        Alternatively, a "chained" style of event handling can be
        used, by configuring the handler with the ``retval=True``
        modifier and returning the new exception instance from the
        function.  In this case, event handling will continue onto the
        next handler.   The "chained" exception is available using
        :attr:`.ExceptionContext.chained_exception`::

            @event.listens_for(Engine, "handle_error", retval=True)
            def handle_exception(context):
                if context.chained_exception is not None and \\
                    "special" in context.chained_exception.message:
                    return MySpecialException("failed",
                        cause=context.chained_exception)

        Handlers that return ``None`` may be used within the chain; when
        a handler returns ``None``, the previous exception instance,
        if any, is maintained as the current exception that is passed onto the
        next handler.

        When a custom exception is raised or returned, SQLAlchemy raises
        this new exception as-is, it is not wrapped by any SQLAlchemy
        object.  If the exception is not a subclass of
        :class:`sqlalchemy.exc.StatementError`,
        certain features may not be available; currently this includes
        the ORM's feature of adding a detail hint about "autoflush" to
        exceptions raised within the autoflush process.

        :param context: an :class:`.ExceptionContext` object.  See this
         class for details on all available members.


        .. seealso::

            :ref:`pool_new_disconnect_codes`

        """

    def do_connect(self, dialect: Dialect, conn_rec: ConnectionPoolEntry, cargs: Tuple[Any, ...], cparams: Dict[str, Any]) -> Optional[DBAPIConnection]:
        """Receive connection arguments before a connection is made.

        This event is useful in that it allows the handler to manipulate the
        cargs and/or cparams collections that control how the DBAPI
        ``connect()`` function will be called. ``cargs`` will always be a
        Python list that can be mutated in-place, and ``cparams`` a Python
        dictionary that may also be mutated::

            e = create_engine("postgresql+psycopg2://user@host/dbname")

            @event.listens_for(e, 'do_connect')
            def receive_do_connect(dialect, conn_rec, cargs, cparams):
                cparams["password"] = "some_password"

        The event hook may also be used to override the call to ``connect()``
        entirely, by returning a non-``None`` DBAPI connection object::

            e = create_engine("postgresql+psycopg2://user@host/dbname")

            @event.listens_for(e, 'do_connect')
            def receive_do_connect(dialect, conn_rec, cargs, cparams):
                return psycopg2.connect(*cargs, **cparams)

        .. seealso::

            :ref:`custom_dbapi_args`

        """

    def do_executemany(self, cursor: DBAPICursor, statement: str, parameters: _DBAPIMultiExecuteParams, context: ExecutionContext) -> Optional[Literal[True]]:
        """Receive a cursor to have executemany() called.

        Return the value True to halt further events from invoking,
        and to indicate that the cursor execution has already taken
        place within the event handler.

        """

    def do_execute_no_params(self, cursor: DBAPICursor, statement: str, context: ExecutionContext) -> Optional[Literal[True]]:
        """Receive a cursor to have execute() with no parameters called.

        Return the value True to halt further events from invoking,
        and to indicate that the cursor execution has already taken
        place within the event handler.

        """

    def do_execute(self, cursor: DBAPICursor, statement: str, parameters: _DBAPISingleExecuteParams, context: ExecutionContext) -> Optional[Literal[True]]:
        """Receive a cursor to have execute() called.

        Return the value True to halt further events from invoking,
        and to indicate that the cursor execution has already taken
        place within the event handler.

        """

    def do_setinputsizes(self, inputsizes: Dict[BindParameter[Any], Any], cursor: DBAPICursor, statement: str, parameters: _DBAPIAnyExecuteParams, context: ExecutionContext) -> None:
        """Receive the setinputsizes dictionary for possible modification.

        This event is emitted in the case where the dialect makes use of the
        DBAPI ``cursor.setinputsizes()`` method which passes information about
        parameter binding for a particular statement.   The given
        ``inputsizes`` dictionary will contain :class:`.BindParameter` objects
        as keys, linked to DBAPI-specific type objects as values; for
        parameters that are not bound, they are added to the dictionary with
        ``None`` as the value, which means the parameter will not be included
        in the ultimate setinputsizes call.   The event may be used to inspect
        and/or log the datatypes that are being bound, as well as to modify the
        dictionary in place.  Parameters can be added, modified, or removed
        from this dictionary.   Callers will typically want to inspect the
        :attr:`.BindParameter.type` attribute of the given bind objects in
        order to make decisions about the DBAPI object.

        After the event, the ``inputsizes`` dictionary is converted into
        an appropriate datastructure to be passed to ``cursor.setinputsizes``;
        either a list for a positional bound parameter execution style,
        or a dictionary of string parameter keys to DBAPI type objects for
        a named bound parameter execution style.

        The setinputsizes hook overall is only used for dialects which include
        the flag ``use_setinputsizes=True``.  Dialects which use this
        include cx_Oracle, pg8000, asyncpg, and pyodbc dialects.

        .. note::

            For use with pyodbc, the ``use_setinputsizes`` flag
            must be passed to the dialect, e.g.::

                create_engine("mssql+pyodbc://...", use_setinputsizes=True)

            .. seealso::

                  :ref:`mssql_pyodbc_setinputsizes`

        .. versionadded:: 1.2.9

        .. seealso::

            :ref:`cx_oracle_setinputsizes`

        """
        pass