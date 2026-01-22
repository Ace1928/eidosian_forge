from __future__ import annotations
from collections.abc import Callable, Mapping, MutableMapping
from typing import TYPE_CHECKING, Any, Optional, Union
from bson.son import SON
from pymongo import common
from pymongo.collation import validate_collation_or_none
from pymongo.errors import ConfigurationError
from pymongo.read_preferences import ReadPreference, _AggWritePref
class _AggregationCommand:
    """The internal abstract base class for aggregation cursors.

    Should not be called directly by application developers. Use
    :meth:`pymongo.collection.Collection.aggregate`, or
    :meth:`pymongo.database.Database.aggregate` instead.
    """

    def __init__(self, target: Union[Database, Collection], cursor_class: type[CommandCursor], pipeline: _Pipeline, options: MutableMapping[str, Any], explicit_session: bool, let: Optional[Mapping[str, Any]]=None, user_fields: Optional[MutableMapping[str, Any]]=None, result_processor: Optional[Callable[[Mapping[str, Any], Connection], None]]=None, comment: Any=None) -> None:
        if 'explain' in options:
            raise ConfigurationError('The explain option is not supported. Use Database.command instead.')
        self._target = target
        pipeline = common.validate_list('pipeline', pipeline)
        self._pipeline = pipeline
        self._performs_write = False
        if pipeline and ('$out' in pipeline[-1] or '$merge' in pipeline[-1]):
            self._performs_write = True
        common.validate_is_mapping('options', options)
        if let is not None:
            common.validate_is_mapping('let', let)
            options['let'] = let
        if comment is not None:
            options['comment'] = comment
        self._options = options
        self._batch_size = common.validate_non_negative_integer_or_none('batchSize', self._options.pop('batchSize', None))
        self._options.setdefault('cursor', {})
        if self._batch_size is not None and (not self._performs_write):
            self._options['cursor']['batchSize'] = self._batch_size
        self._cursor_class = cursor_class
        self._explicit_session = explicit_session
        self._user_fields = user_fields
        self._result_processor = result_processor
        self._collation = validate_collation_or_none(options.pop('collation', None))
        self._max_await_time_ms = options.pop('maxAwaitTimeMS', None)
        self._write_preference: Optional[_AggWritePref] = None

    @property
    def _aggregation_target(self) -> Union[str, int]:
        """The argument to pass to the aggregate command."""
        raise NotImplementedError

    @property
    def _cursor_namespace(self) -> str:
        """The namespace in which the aggregate command is run."""
        raise NotImplementedError

    def _cursor_collection(self, cursor_doc: Mapping[str, Any]) -> Collection:
        """The Collection used for the aggregate command cursor."""
        raise NotImplementedError

    @property
    def _database(self) -> Database:
        """The database against which the aggregation command is run."""
        raise NotImplementedError

    def get_read_preference(self, session: Optional[ClientSession]) -> Union[_AggWritePref, _ServerMode]:
        if self._write_preference:
            return self._write_preference
        pref = self._target._read_preference_for(session)
        if self._performs_write and pref != ReadPreference.PRIMARY:
            self._write_preference = pref = _AggWritePref(pref)
        return pref

    def get_cursor(self, session: Optional[ClientSession], server: Server, conn: Connection, read_preference: _ServerMode) -> CommandCursor[_DocumentType]:
        cmd = SON([('aggregate', self._aggregation_target), ('pipeline', self._pipeline)])
        cmd.update(self._options)
        if 'readConcern' not in cmd and (not self._performs_write or conn.max_wire_version >= 8):
            read_concern = self._target.read_concern
        else:
            read_concern = None
        if 'writeConcern' not in cmd and self._performs_write:
            write_concern = self._target._write_concern_for(session)
        else:
            write_concern = None
        result = conn.command(self._database.name, cmd, read_preference, self._target.codec_options, parse_write_concern_error=True, read_concern=read_concern, write_concern=write_concern, collation=self._collation, session=session, client=self._database.client, user_fields=self._user_fields)
        if self._result_processor:
            self._result_processor(result, conn)
        if 'cursor' in result:
            cursor = result['cursor']
        else:
            cursor = {'id': 0, 'firstBatch': result.get('result', []), 'ns': self._cursor_namespace}
        cmd_cursor = self._cursor_class(self._cursor_collection(cursor), cursor, conn.address, batch_size=self._batch_size or 0, max_await_time_ms=self._max_await_time_ms, session=session, explicit_session=self._explicit_session, comment=self._options.get('comment'))
        cmd_cursor._maybe_pin_connection(conn)
        return cmd_cursor