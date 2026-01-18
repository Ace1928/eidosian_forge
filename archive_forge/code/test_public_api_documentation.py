import pkgutil
import types
import importlib
import warnings
from importlib import import_module
import pytest
import scipy

    Check that all submodules listed higher up in this file can be imported
    Note that if a PRIVATE_BUT_PRESENT_MODULES entry goes missing, it may
    simply need to be removed from the list (deprecation may or may not be
    needed - apply common sense).
    