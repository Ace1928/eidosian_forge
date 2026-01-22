import logging
import pathlib
import json
import random
import string
import socket
import os
import threading
from typing import Dict, Optional
from datetime import datetime
from google.protobuf.json_format import Parse
from ray.core.generated.event_pb2 import Event
from ray._private.protobuf_compat import message_to_dict
class EventLoggerAdapter:

    def __init__(self, source: Event.SourceType, logger: logging.Logger):
        """Adapter for the Python logger that's used to emit events.

        When events are emitted, they are aggregated and available via
        state API and dashboard.

        This class is thread-safe.
        """
        self.logger = logger
        self.source = source
        self.source_hostname = socket.gethostname()
        self.source_pid = os.getpid()
        self.lock = threading.Lock()
        self.global_context = {}

    def set_global_context(self, global_context: Dict[str, str]=None):
        """Set the global metadata.

        This method overwrites the global metadata if it is called more than once.
        """
        with self.lock:
            self.global_context = {} if not global_context else global_context

    def trace(self, message: str, **kwargs):
        self._emit(Event.Severity.TRACE, message, **kwargs)

    def debug(self, message: str, **kwargs):
        self._emit(Event.Severity.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs):
        self._emit(Event.Severity.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs):
        self._emit(Event.Severity.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs):
        self._emit(Event.Severity.ERROR, message, **kwargs)

    def fatal(self, message: str, **kwargs):
        self._emit(Event.Severity.FATAL, message, **kwargs)

    def _emit(self, severity: Event.Severity, message: str, **kwargs):
        event = Event()
        event.event_id = get_event_id()
        event.timestamp = int(datetime.now().timestamp())
        event.message = message
        event.severity = severity
        event.label = ''
        event.source_type = self.source
        event.source_hostname = self.source_hostname
        event.source_pid = self.source_pid
        custom_fields = event.custom_fields
        with self.lock:
            for k, v in self.global_context.items():
                if v is not None and k is not None:
                    custom_fields[k] = v
        for k, v in kwargs.items():
            if v is not None and k is not None:
                custom_fields[k] = v
        self.logger.info(json.dumps(message_to_dict(event, always_print_fields_with_no_presence=True, preserving_proto_field_name=True, use_integers_for_enums=False)))
        self.logger.handlers[0].flush()