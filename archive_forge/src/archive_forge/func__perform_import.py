from collections.abc import Mapping
import inspect
import importlib
import logging
import sys
import warnings
from .deprecation import deprecated, deprecation_warning, in_testing_environment
from .errors import DeferredImportError
def _perform_import(name, error_message, minimum_version, callback, importer, catch_exceptions, package):
    import_error = None
    version_error = None
    try:
        with warnings.catch_warnings():
            if SUPPRESS_DEPENDENCY_WARNINGS and (not name.startswith('pyomo.')):
                warnings.resetwarnings()
                warnings.simplefilter('ignore')
            if importer is None:
                module = importlib.import_module(name)
            else:
                module = importer()
        if minimum_version is None or check_min_version(module, minimum_version):
            if callback is not None:
                callback(module, True)
            return (module, True)
        else:
            version = getattr(module, '__version__', 'UNKNOWN')
            version_error = 'version %s does not satisfy the minimum version %s' % (version, minimum_version)
    except catch_exceptions as e:
        import_error = '%s: %s' % (type(e).__name__, e)
    module = ModuleUnavailable(name, error_message, version_error, import_error, package)
    if callback is not None:
        callback(module, False)
    return (module, False)