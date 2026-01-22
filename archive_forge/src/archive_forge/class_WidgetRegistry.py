import os
import sys
import typing
from contextlib import contextmanager
from collections.abc import Iterable
from IPython import get_ipython
from traitlets import (
from json import loads as jsonloads, dumps as jsondumps
from .. import comm
from base64 import standard_b64encode
from .utils import deprecation, _get_frame
from .._version import __protocol_version__, __control_protocol_version__, __jupyter_widgets_base_version__
import inspect
class WidgetRegistry:

    def __init__(self):
        self._registry = {}

    def register(self, model_module, model_module_version_range, model_name, view_module, view_module_version_range, view_name, klass):
        """Register a value"""
        model_module = self._registry.setdefault(model_module, {})
        model_version = model_module.setdefault(model_module_version_range, {})
        model_name = model_version.setdefault(model_name, {})
        view_module = model_name.setdefault(view_module, {})
        view_version = view_module.setdefault(view_module_version_range, {})
        view_version[view_name] = klass

    def get(self, model_module, model_module_version, model_name, view_module, view_module_version, view_name):
        """Get a value"""
        module_versions = self._registry[model_module]
        model_names = list(module_versions.values())[0]
        view_modules = model_names[model_name]
        view_versions = view_modules[view_module]
        view_names = list(view_versions.values())[0]
        widget_class = view_names[view_name]
        return widget_class

    def items(self):
        for model_module, mm in sorted(self._registry.items()):
            for model_version, mv in sorted(mm.items()):
                for model_name, vm in sorted(mv.items()):
                    for view_module, vv in sorted(vm.items()):
                        for view_version, vn in sorted(vv.items()):
                            for view_name, widget in sorted(vn.items()):
                                yield ((model_module, model_version, model_name, view_module, view_version, view_name), widget)