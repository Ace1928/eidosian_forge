def _requires_builtin_wrapper(self, fullname):
    if fullname not in sys.builtin_module_names:
        raise ImportError('{!r} is not a built-in module'.format(fullname), name=fullname)
    return fxn(self, fullname)