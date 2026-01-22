from __future__ import annotations
import asyncio
import copy
import json
import logging
import typing as t
import warnings
from datetime import datetime, timezone
from jsonschema import ValidationError
from pythonjsonlogger import jsonlogger
from traitlets import Dict, Instance, Set, default
from traitlets.config import Config, LoggingConfigurable
from .schema import SchemaType
from .schema_registry import SchemaRegistry
from .traits import Handlers
from .validators import JUPYTER_EVENTS_CORE_VALIDATOR
class EventLogger(LoggingConfigurable):
    """
    An Event logger for emitting structured events.

    Event schemas must be registered with the
    EventLogger using the `register_schema` or
    `register_schema_file` methods. Every schema
    will be validated against Jupyter Event's metaschema.
    """
    handlers = Handlers(default_value=None, allow_none=True, help='A list of logging.Handler instances to send events to.\n\n        When set to None (the default), all events are discarded.\n        ').tag(config=True)
    schemas = Instance(SchemaRegistry, help='The SchemaRegistry for caching validated schemas\n        and their jsonschema validators.\n        ')
    _modifiers = Dict({}, help='A mapping of schemas to their list of modifiers.')
    _modified_listeners = Dict({}, help='A mapping of schemas to the listeners of modified events.')
    _unmodified_listeners = Dict({}, help='A mapping of schemas to the listeners of unmodified/raw events.')
    _active_listeners: set[asyncio.Task[t.Any]] = Set()

    async def gather_listeners(self) -> list[t.Any]:
        """Gather all of the active listeners."""
        return await asyncio.gather(*self._active_listeners, return_exceptions=True)

    @default('schemas')
    def _default_schemas(self) -> SchemaRegistry:
        return SchemaRegistry()

    def __init__(self, *args: t.Any, **kwargs: t.Any) -> None:
        """Initialize the logger."""
        super().__init__(*args, **kwargs)
        log_name = __name__ + '.' + str(id(self))
        self._logger = logging.getLogger(log_name)
        self._logger.propagate = False
        self._logger.setLevel(logging.INFO)
        if self.handlers:
            for handler in self.handlers:
                self.register_handler(handler)

    def _load_config(self, cfg: Config, section_names: list[str] | None=None, traits: list[str] | None=None) -> None:
        """Load EventLogger traits from a Config object, patching the
        handlers trait in the Config object to avoid deepcopy errors.
        """
        my_cfg = self._find_my_config(cfg)
        handlers: list[logging.Handler] = my_cfg.pop('handlers', [])

        def get_handlers() -> list[logging.Handler]:
            return handlers
        my_cfg['handlers'] = get_handlers
        eventlogger_cfg = Config({'EventLogger': my_cfg})
        super()._load_config(eventlogger_cfg, section_names=None, traits=None)

    def register_event_schema(self, schema: SchemaType) -> None:
        """Register this schema with the schema registry.

        Get this registered schema using the EventLogger.schema.get() method.
        """
        event_schema = self.schemas.register(schema)
        key = event_schema.id
        if key not in self._modifiers:
            self._modifiers[key] = set()
        if key not in self._modified_listeners:
            self._modified_listeners[key] = set()
        if key not in self._unmodified_listeners:
            self._unmodified_listeners[key] = set()

    def register_handler(self, handler: logging.Handler) -> None:
        """Register a new logging handler to the Event Logger.

        All outgoing messages will be formatted as a JSON string.
        """

        def _handle_message_field(record: t.Any, **kwargs: t.Any) -> str:
            """Python's logger always emits the "message" field with
            the value as "null" unless it's present in the schema/data.
            Message happens to be a common field for event logs,
            so special case it here and only emit it if "message"
            is found the in the schema's property list.
            """
            schema = self.schemas.get(record['__schema__'])
            if 'message' not in schema.properties:
                del record['message']
            return json.dumps(record, **kwargs)
        formatter = jsonlogger.JsonFormatter(json_serializer=_handle_message_field)
        handler.setFormatter(formatter)
        self._logger.addHandler(handler)
        if handler not in self.handlers:
            self.handlers.append(handler)

    def remove_handler(self, handler: logging.Handler) -> None:
        """Remove a logging handler from the logger and list of handlers."""
        self._logger.removeHandler(handler)
        if handler in self.handlers:
            self.handlers.remove(handler)

    def add_modifier(self, *, schema_id: str | None=None, modifier: t.Callable[[str, dict[str, t.Any]], dict[str, t.Any]]) -> None:
        """Add a modifier (callable) to a registered event.

        Parameters
        ----------
        modifier: Callable
            A callable function/method that executes when the named event occurs.
            This method enforces a string signature for modifiers:

                (schema_id: str, data: dict) -> dict:
        """
        if not callable(modifier):
            msg = '`modifier` must be a callable'
            raise TypeError(msg)
        if schema_id:
            modifiers = self._modifiers.get(schema_id, set())
            modifiers.add(modifier)
            self._modifiers[schema_id] = modifiers
            return
        for id_ in self._modifiers:
            if schema_id is None or id_ == schema_id:
                self._modifiers[id_].add(modifier)

    def remove_modifier(self, *, schema_id: str | None=None, modifier: t.Callable[[str, dict[str, t.Any]], dict[str, t.Any]]) -> None:
        """Remove a modifier from an event or all events.

        Parameters
        ----------
        schema_id: str
            If given, remove this modifier only for a specific event type.
        modifier: Callable[[str, dict], dict]

            The modifier to remove.
        """
        if schema_id:
            self._modifiers[schema_id].discard(modifier)
        else:
            for schema_id in self.schemas.schema_ids:
                self._modifiers[schema_id].discard(modifier)
                self._modifiers[schema_id].discard(modifier)

    def add_listener(self, *, modified: bool=True, schema_id: str | None=None, listener: t.Callable[[EventLogger, str, dict[str, t.Any]], t.Coroutine[t.Any, t.Any, None]]) -> None:
        """Add a listener (callable) to a registered event.

        Parameters
        ----------
        modified: bool
            If True (default), listens to the data after it has been mutated/modified
            by the list of modifiers.
        schema_id: str
            $id of the schema
        listener: Callable
            A callable function/method that executes when the named event occurs.
        """
        if not callable(listener):
            msg = '`listener` must be a callable'
            raise TypeError(msg)
        if schema_id:
            if modified:
                listeners = self._modified_listeners.get(schema_id, set())
                listeners.add(listener)
                self._modified_listeners[schema_id] = listeners
                return
            listeners = self._unmodified_listeners.get(schema_id, set())
            listeners.add(listener)
            self._unmodified_listeners[schema_id] = listeners
            return
        for id_ in self.schemas.schema_ids:
            if schema_id is None or id_ == schema_id:
                if modified:
                    self._modified_listeners[id_].add(listener)
                else:
                    self._unmodified_listeners[id_].add(listener)

    def remove_listener(self, *, schema_id: str | None=None, listener: t.Callable[[EventLogger, str, dict[str, t.Any]], t.Coroutine[t.Any, t.Any, None]]) -> None:
        """Remove a listener from an event or all events.

        Parameters
        ----------
        schema_id: str
            If given, remove this modifier only for a specific event type.

        listener: Callable[[EventLogger, str, dict], dict]
            The modifier to remove.
        """
        if schema_id:
            self._modified_listeners[schema_id].discard(listener)
            self._unmodified_listeners[schema_id].discard(listener)
        else:
            for schema_id in self.schemas.schema_ids:
                self._modified_listeners[schema_id].discard(listener)
                self._unmodified_listeners[schema_id].discard(listener)

    def emit(self, *, schema_id: str, data: dict[str, t.Any], timestamp_override: datetime | None=None) -> dict[str, t.Any] | None:
        """
        Record given event with schema has occurred.

        Parameters
        ----------
        schema_id: str
            $id of the schema
        data: dict
            The event to record
        timestamp_override: datetime, optional
            Optionally override the event timestamp. By default it is set to the current timestamp.

        Returns
        -------
        dict
            The recorded event data
        """
        if not self.handlers and (not self._modified_listeners[schema_id]) and (not self._unmodified_listeners[schema_id]):
            return None
        if schema_id not in self.schemas:
            warnings.warn(f'{schema_id} has not been registered yet. If this was not intentional, please register the schema using the `register_event_schema` method.', SchemaNotRegistered, stacklevel=2)
            return None
        schema = self.schemas.get(schema_id)
        modified_data = copy.deepcopy(data)
        for modifier in self._modifiers[schema.id]:
            modified_data = modifier(schema_id=schema_id, data=modified_data)
        if self._unmodified_listeners[schema.id]:
            self.schemas.validate_event(schema_id, data)
        self.schemas.validate_event(schema_id, modified_data)
        timestamp = datetime.now(tz=timezone.utc) if timestamp_override is None else timestamp_override
        capsule = {'__timestamp__': timestamp.isoformat() + 'Z', '__schema__': schema_id, '__schema_version__': schema.version, '__metadata_version__': EVENTS_METADATA_VERSION}
        try:
            JUPYTER_EVENTS_CORE_VALIDATOR.validate(capsule)
        except ValidationError as err:
            raise CoreMetadataError from err
        capsule.update(modified_data)
        self._logger.info(capsule)

        def _listener_task_done(task: asyncio.Task[t.Any]) -> None:
            err = task.exception()
            if err:
                self.log.error(err)
            self._active_listeners.discard(task)
        for listener in self._modified_listeners[schema_id]:
            task = asyncio.create_task(listener(logger=self, schema_id=schema_id, data=modified_data))
            self._active_listeners.add(task)
            task.add_done_callback(_listener_task_done)
        for listener in self._unmodified_listeners[schema_id]:
            task = asyncio.create_task(listener(logger=self, schema_id=schema_id, data=data))
            self._active_listeners.add(task)

            def _listener_task_done(task: asyncio.Task[t.Any]) -> None:
                err = task.exception()
                if err:
                    self.log.error(err)
                self._active_listeners.discard(task)
            task.add_done_callback(_listener_task_done)
        return capsule