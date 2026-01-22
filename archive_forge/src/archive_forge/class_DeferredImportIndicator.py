from collections.abc import Mapping
import inspect
import importlib
import logging
import sys
import warnings
from .deprecation import deprecated, deprecation_warning, in_testing_environment
from .errors import DeferredImportError
class DeferredImportIndicator(_DeferredImportIndicatorBase):
    """Placeholder indicating if an import was successful.

    This object serves as a placeholder for the Boolean indicator if a
    deferred module import was successful.  Casting this instance to
    `bool` will cause the import to be attempted.  The actual import logic
    is here and not in the :py:class:`DeferredImportModule` to reduce the number of
    attributes on the :py:class:`DeferredImportModule`.

    :py:class:`DeferredImportIndicator` supports limited logical expressions
    using the ``&`` (and) and ``|`` (or) binary operators.  Creating
    these expressions does not trigger the import of the corresponding
    :py:class:`DeferredImportModule` instances, although casting the
    resulting expression to ``bool()`` will trigger any relevant
    imports.

    """

    def __init__(self, name, error_message, catch_exceptions, minimum_version, original_globals, callback, importer, deferred_submodules):
        self._names = [name]
        for _n in tuple(self._names):
            if '.' in _n:
                self._names.append(_n.split('.')[-1])
        self._error_message = error_message
        self._catch_exceptions = catch_exceptions
        self._minimum_version = minimum_version
        self._original_globals = original_globals
        self._callback = callback
        self._importer = importer
        self._module = None
        self._available = None
        self._deferred_submodules = deferred_submodules

    def __bool__(self):
        self.resolve()
        return self._available

    def resolve(self):
        if self._module is None:
            package = self._original_globals.get('__name__', '')
            try:
                self._module, self._available = _perform_import(name=self._names[0], error_message=self._error_message, minimum_version=self._minimum_version, callback=self._callback, importer=self._importer, catch_exceptions=self._catch_exceptions, package=package)
            except Exception as e:
                self._module = ModuleUnavailable(self._names[0], 'Exception raised when importing %s' % (self._names[0],), None, '%s: %s' % (type(e).__name__, e), package)
                self._available = False
                raise
            if self._deferred_submodules and type(self._module) is ModuleUnavailable:
                info = self._module._moduleunavailable_info_
                for submod in self._deferred_submodules:
                    refmod = self._module
                    for name in submod.split('.')[1:]:
                        try:
                            refmod = getattr(refmod, name)
                        except DeferredImportError:
                            setattr(refmod, name, ModuleUnavailable(refmod.__name__ + submod, *info))
                            refmod = getattr(refmod, name)
            self.replace_self_in_globals(self._original_globals)
        _frame = inspect.currentframe().f_back
        while _frame.f_globals is globals():
            _frame = _frame.f_back
        self.replace_self_in_globals(_frame.f_globals)

    def replace_self_in_globals(self, _globals):
        for k, v in _globals.items():
            if v is self:
                _globals[k] = self._available
            elif v.__class__ is DeferredImportModule and v._indicator_flag is self:
                if v._submodule_name is None:
                    _globals[k] = self._module
                else:
                    _mod_path = v._submodule_name.split('.')[1:]
                    _mod = self._module
                    for _sub in _mod_path:
                        _mod = getattr(_mod, _sub)
                    _globals[k] = _mod