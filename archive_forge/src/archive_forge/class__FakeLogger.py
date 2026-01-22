import collections
import logging
import pytest
import modin.logging
from modin.config import LogMode
class _FakeLogger:
    _loggers = {}

    def __init__(self, namespace):
        self.messages = collections.defaultdict(list)
        self.namespace = namespace

    def log(self, log_level, message, *args, **kw):
        self.messages[log_level].append(message.format(*args, **kw))

    def exception(self, message, *args, **kw):
        self.messages['exception'].append(message.format(*args, **kw))

    @classmethod
    def make(cls, namespace):
        return cls._loggers.setdefault(namespace, cls(namespace))

    @classmethod
    def get(cls, namespace='modin.logger.default'):
        return cls._loggers[namespace].messages

    @classmethod
    def clear(cls):
        cls._loggers = {}