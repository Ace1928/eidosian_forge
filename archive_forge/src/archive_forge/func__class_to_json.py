import os
from traitlets import Any, Unicode, List, Dict, Union, Instance
from ipywidgets import DOMWidget
from ipywidgets.widgets.widget import widget_serialization
from .Template import Template, get_template
from ._version import semver
from .ForceLoad import force_load_instance
import inspect
from importlib import import_module
def _class_to_json(x, obj):
    if not x:
        return widget_serialization['to_json'](x, obj)
    return {k: _value_to_json(v, obj) for k, v in x.items()}