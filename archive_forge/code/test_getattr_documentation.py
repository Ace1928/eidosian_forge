from importlib import import_module
from pkgutil import walk_packages
import matplotlib
import pytest

    Test that __getattr__ methods raise AttributeError for unknown keys.
    See #20822, #20855.
    