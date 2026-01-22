import collections
import copy
import inspect
import logging
import pkgutil
import sys
import types
def _true(*args, **kwargs):
    """Return ``True``."""
    return True