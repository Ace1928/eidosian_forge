import os
import weakref
import torch
def _add_script_class(python_class, script_class):
    _script_classes[python_class] = script_class
    _name_to_pyclass[script_class.qualified_name()] = python_class