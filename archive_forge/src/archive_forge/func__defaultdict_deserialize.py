import dataclasses
import importlib
import json
import threading
import warnings
from collections import defaultdict, deque, namedtuple, OrderedDict
from typing import (
def _defaultdict_deserialize(dumpable_context: DumpableContext) -> Context:
    assert isinstance(dumpable_context, dict)
    assert set(dumpable_context) == {'default_factory_module', 'default_factory_name', 'dict_context'}
    default_factory_module = dumpable_context['default_factory_module']
    default_factory_name = dumpable_context['default_factory_name']
    assert isinstance(default_factory_module, str)
    assert isinstance(default_factory_name, str)
    module = importlib.import_module(default_factory_module)
    default_factory = getattr(module, default_factory_name)
    dict_context = dumpable_context['dict_context']
    return [default_factory, dict_context]