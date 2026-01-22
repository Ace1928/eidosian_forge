from functools import partial
import itertools
import os
import sys
import socket as socket_module
from _pydev_bundle._pydev_imports_tipper import TYPE_IMPORT, TYPE_CLASS, TYPE_FUNCTION, TYPE_ATTR, \
from _pydev_bundle.pydev_is_thread_alive import is_thread_alive
from _pydev_bundle.pydev_override import overrides
from _pydevd_bundle._debug_adapter import pydevd_schema
from _pydevd_bundle._debug_adapter.pydevd_schema import ModuleEvent, ModuleEventBody, Module, \
from _pydevd_bundle.pydevd_comm_constants import CMD_THREAD_CREATE, CMD_RETURN, CMD_MODULE_EVENT, \
from _pydevd_bundle.pydevd_constants import get_thread_id, ForkSafeLock, DebugInfoHolder
from _pydevd_bundle.pydevd_net_command import NetCommand, NULL_NET_COMMAND
from _pydevd_bundle.pydevd_net_command_factory_xml import NetCommandFactory
from _pydevd_bundle.pydevd_utils import get_non_pydevd_threads
import pydevd_file_utils
from _pydevd_bundle.pydevd_comm import build_exception_info_response
from _pydevd_bundle.pydevd_additional_thread_info import set_additional_thread_info
from _pydevd_bundle import pydevd_frame_utils, pydevd_constants, pydevd_utils
import linecache
from io import StringIO
from _pydev_bundle import pydev_log
class ModulesManager(object):

    def __init__(self):
        self._lock = ForkSafeLock()
        self._modules = {}
        self._next_id = partial(next, itertools.count(0))

    def track_module(self, filename_in_utf8, module_name, frame):
        """
        :return list(NetCommand):
            Returns a list with the module events to be sent.
        """
        if filename_in_utf8 in self._modules:
            return []
        module_events = []
        with self._lock:
            if filename_in_utf8 in self._modules:
                return
            try:
                version = str(frame.f_globals.get('__version__', ''))
            except:
                version = '<unknown>'
            try:
                package_name = str(frame.f_globals.get('__package__', ''))
            except:
                package_name = '<unknown>'
            module_id = self._next_id()
            module = Module(module_id, module_name, filename_in_utf8)
            if version:
                module.version = version
            if package_name:
                module.kwargs['package'] = package_name
            module_event = ModuleEvent(ModuleEventBody('new', module))
            module_events.append(NetCommand(CMD_MODULE_EVENT, 0, module_event, is_json=True))
            self._modules[filename_in_utf8] = module.to_dict()
        return module_events

    def get_modules_info(self):
        """
        :return list(Module)
        """
        with self._lock:
            return list(self._modules.values())