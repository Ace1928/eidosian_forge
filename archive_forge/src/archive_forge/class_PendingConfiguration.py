import inspect
import os
import sys
import threading
import warnings
from collections import UserDict, defaultdict, deque
from datetime import datetime
from datetime import timezone as datetime_timezone
from operator import attrgetter
from click.exceptions import Exit
from dateutil.parser import isoparse
from kombu import pools
from kombu.clocks import LamportClock
from kombu.common import oid_from
from kombu.utils.compat import register_after_fork
from kombu.utils.objects import cached_property
from kombu.utils.uuid import uuid
from vine import starpromise
from celery import platforms, signals
from celery._state import (_announce_app_finalized, _deregister_app, _register_app, _set_current_app, _task_stack,
from celery.exceptions import AlwaysEagerIgnored, ImproperlyConfigured
from celery.loaders import get_loader_cls
from celery.local import PromiseProxy, maybe_evaluate
from celery.utils import abstract
from celery.utils.collections import AttributeDictMixin
from celery.utils.dispatch import Signal
from celery.utils.functional import first, head_from_fun, maybe_list
from celery.utils.imports import gen_task_name, instantiate, symbol_by_name
from celery.utils.log import get_logger
from celery.utils.objects import FallbackContext, mro_lookup
from celery.utils.time import maybe_make_aware, timezone, to_utc
from . import backends, builtins  # noqa
from .annotations import prepare as prepare_annotations
from .autoretry import add_autoretry_behaviour
from .defaults import DEFAULT_SECURITY_DIGEST, find_deprecated_settings
from .registry import TaskRegistry
from .utils import (AppPickler, Settings, _new_key_to_old, _old_key_to_new, _unpickle_app, _unpickle_app_v2, appstr,
class PendingConfiguration(UserDict, AttributeDictMixin):
    callback = None
    _data = None

    def __init__(self, conf, callback):
        object.__setattr__(self, '_data', conf)
        object.__setattr__(self, 'callback', callback)

    def __setitem__(self, key, value):
        self._data[key] = value

    def clear(self):
        self._data.clear()

    def update(self, *args, **kwargs):
        self._data.update(*args, **kwargs)

    def setdefault(self, *args, **kwargs):
        return self._data.setdefault(*args, **kwargs)

    def __contains__(self, key):
        return key in self._data

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return repr(self.data)

    @cached_property
    def data(self):
        return self.callback()