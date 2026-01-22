import os
import weakref
import torch
def _clear_class_state():
    _script_classes.clear()
    _name_to_pyclass.clear()