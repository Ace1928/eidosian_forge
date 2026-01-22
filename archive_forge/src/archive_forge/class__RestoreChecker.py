from lxml import etree
import sys
import re
import doctest
class _RestoreChecker:

    def __init__(self, dt_self, old_checker, new_checker, check_func, clone_func, del_module):
        self.dt_self = dt_self
        self.checker = old_checker
        self.checker._temp_call_super_check_output = self.call_super
        self.checker._temp_override_self = new_checker
        self.check_func = check_func
        self.clone_func = clone_func
        self.del_module = del_module
        self.install_clone()
        self.install_dt_self()

    def install_clone(self):
        self.func_code = self.check_func.__code__
        self.func_globals = self.check_func.__globals__
        self.check_func.__code__ = self.clone_func.__code__

    def uninstall_clone(self):
        self.check_func.__code__ = self.func_code

    def install_dt_self(self):
        self.prev_func = self.dt_self._DocTestRunner__record_outcome
        self.dt_self._DocTestRunner__record_outcome = self

    def uninstall_dt_self(self):
        self.dt_self._DocTestRunner__record_outcome = self.prev_func

    def uninstall_module(self):
        if self.del_module:
            import sys
            del sys.modules[self.del_module]
            if '.' in self.del_module:
                package, module = self.del_module.rsplit('.', 1)
                package_mod = sys.modules[package]
                delattr(package_mod, module)

    def __call__(self, *args, **kw):
        self.uninstall_clone()
        self.uninstall_dt_self()
        del self.checker._temp_override_self
        del self.checker._temp_call_super_check_output
        result = self.prev_func(*args, **kw)
        self.uninstall_module()
        return result

    def call_super(self, *args, **kw):
        self.uninstall_clone()
        try:
            return self.check_func(*args, **kw)
        finally:
            self.install_clone()