from concurrent import futures
import enum
import errno
import io
import logging as pylogging
import os
import platform
import socket
import subprocess
import sys
import tempfile
import threading
import eventlet
from eventlet import patcher
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import encodeutils
from oslo_utils import importutils
from oslo_privsep._i18n import _
from oslo_privsep import capabilities
from oslo_privsep import comm
def un_monkey_patch():
    for eventlet_mod_name, func_modules in EVENTLET_LIBRARIES:
        if not eventlet.patcher.is_monkey_patched(eventlet_mod_name):
            continue
        for name, mod in func_modules():
            patched_mod = sys.modules.get(name)
            orig_mod = eventlet.patcher.original(name)
            for attr_name in mod.__patched__:
                patched_attr = getattr(mod, attr_name, None)
                unpatched_attr = getattr(orig_mod, attr_name, None)
                if patched_attr is not None:
                    setattr(patched_mod, attr_name, unpatched_attr)