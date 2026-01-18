import atexit
import os
import sys
import __main__
from contextlib import suppress
from io import BytesIO
import dill
import json                                         # top-level module
import urllib as url                                # top-level module under alias
from xml import sax                                 # submodule
import xml.dom.minidom as dom                       # submodule under alias
import test_dictviews as local_mod                  # non-builtin top-level module
from calendar import Calendar, isleap, day_name     # class, function, other object
from cmath import log as complex_log                # imported with alias
def test_session_other():
    """test dump/load_module() for a module other than __main__"""
    import test_classdef as module
    atexit.register(_clean_up_cache, module)
    module.selfref = module
    dict_objects = [obj for obj in module.__dict__.keys() if not obj.startswith('__')]
    session_buffer = BytesIO()
    dill.dump_module(session_buffer, module)
    for obj in dict_objects:
        del module.__dict__[obj]
    session_buffer.seek(0)
    dill.load_module(session_buffer, module)
    assert all((obj in module.__dict__ for obj in dict_objects))
    assert module.selfref is module