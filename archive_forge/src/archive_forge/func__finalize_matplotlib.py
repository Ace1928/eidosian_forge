from collections.abc import Mapping
import inspect
import importlib
import logging
import sys
import warnings
from .deprecation import deprecated, deprecation_warning, in_testing_environment
from .errors import DeferredImportError
def _finalize_matplotlib(module, available):
    if not available:
        return
    if in_testing_environment():
        module.use('Agg')
    import matplotlib.pyplot