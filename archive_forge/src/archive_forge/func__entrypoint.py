import copy
import enum
import functools
import logging
import multiprocessing
import shlex
import sys
import threading
from oslo_config import cfg
from oslo_config import types
from oslo_utils import importutils
from oslo_privsep._i18n import _
from oslo_privsep import capabilities
from oslo_privsep import daemon
def _entrypoint(self, func):
    if not func.__module__.startswith(self.prefix):
        raise AssertionError('%r entrypoints must be below "%s"' % (self, self.prefix))
    if getattr(func, _ENTRYPOINT_ATTR, None) is not None:
        raise AssertionError('%r is already associated with another PrivContext' % func)
    f = functools.partial(self._wrap, func)
    setattr(f, _ENTRYPOINT_ATTR, self)
    return f