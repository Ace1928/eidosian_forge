import sys
def try_imports(module_names, alternative=_RAISE_EXCEPTION, error_callback=None):
    """Attempt to import modules.

    Tries to import the first module in ``module_names``.  If it can be
    imported, we return it.  If not, we go on to the second module and try
    that.  The process continues until we run out of modules to try.  If none
    of the modules can be imported, either raise an exception or return the
    provided ``alternative`` value.

    :param module_names: A sequence of module names to try to import.
    :param alternative: The value to return if no module can be imported.
        If unspecified, we raise an ImportError.
    :param error_callback: If None, called with the ImportError for *each*
        module that fails to load.
    :raises ImportError: If none of the modules can be imported and no
        alternative value was specified.
    """
    module_names = list(module_names)
    for module_name in module_names:
        module = try_import(module_name, error_callback=error_callback)
        if module:
            return module
    if alternative is _RAISE_EXCEPTION:
        raise ImportError('Could not import any of: %s' % ', '.join(module_names))
    return alternative