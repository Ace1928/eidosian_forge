import copy
import eventlet
import fixtures
import functools
import logging as pylogging
import platform
import sys
import time
from unittest import mock
from oslo_log import formatters
from oslo_log import log as logging
from oslotest import base
import testtools
from oslo_privsep import capabilities
from oslo_privsep import comm
from oslo_privsep import daemon
from oslo_privsep.tests import testctx
def test_un_monkey_patch(self):
    self.assertFalse(any((eventlet.patcher.is_monkey_patched(eventlet_mod_name) for eventlet_mod_name in daemon.EVENTLET_MODULES)))
    eventlet.monkey_patch()
    self.assertTrue(any((eventlet.patcher.is_monkey_patched(eventlet_mod_name) for eventlet_mod_name in daemon.EVENTLET_MODULES)))
    daemon.un_monkey_patch()
    for eventlet_mod_name, func_modules in daemon.EVENTLET_LIBRARIES:
        if not eventlet.patcher.is_monkey_patched(eventlet_mod_name):
            continue
        for name, green_mod in func_modules():
            orig_mod = eventlet.patcher.original(name)
            patched_mod = sys.modules.get(name)
            for attr_name in green_mod.__patched__:
                un_monkey_patched_attr = getattr(patched_mod, attr_name, None)
                original_attr = getattr(orig_mod, attr_name, None)
                self.assertEqual(un_monkey_patched_attr, original_attr)