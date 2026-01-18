from __future__ import annotations
def tracking_import(module, globals=globals(), locals=[], fromlist=None, level=-1):
    """__import__ wrapper - does not change imports at all, but tracks them.

        Default order is implemented by doing output directly.
        All other orders are implemented by collecting output information into
        a sorted list that will be emitted after all imports are processed.

        Indirect imports can only occur after the requested symbol has been
        imported directly (because the indirect import would not have a module
        to pick the symbol up from).
        So this code detects indirect imports by checking whether the symbol in
        question was already imported.

        Keeps the semantics of __import__ unchanged."""
    global options, symbol_definers
    caller_frame = inspect.getframeinfo(sys._getframe(1))
    importer_filename = caller_frame.filename
    importer_module = globals['__name__']
    if importer_filename == caller_frame.filename:
        importer_reference = '%s line %s' % (importer_filename, str(caller_frame.lineno))
    else:
        importer_reference = importer_filename
    result = builtin_import(module, globals, locals, fromlist, level)
    importee_module = result.__name__
    if relevant(importer_module) and relevant(importee_module):
        for symbol in result.__dict__.iterkeys():
            definition = Definition(symbol, result.__dict__[symbol], importer_module)
            if definition not in symbol_definers:
                symbol_definers[definition] = importee_module
        if hasattr(result, '__path__'):
            if options.by_origin:
                msg('Error: %s (a package) is imported by %s', module, importer_reference)
            else:
                msg('Error: %s contains package import %s', importer_reference, module)
        if fromlist != None:
            symbol_list = fromlist
            if '*' in symbol_list:
                if importer_filename.endswith('__init__.py') or importer_filename.endswith('__init__.pyc') or importer_filename.endswith('__init__.pyo'):
                    symbol_list = []
                else:
                    symbol_list = result.__dict__.iterkeys()
            for symbol in symbol_list:
                if symbol not in result.__dict__:
                    if options.by_origin:
                        msg('Error: %s.%s is not defined (yet), but %s tries to import it', importee_module, symbol, importer_reference)
                    else:
                        msg('Error: %s tries to import %s.%s, which did not define it (yet)', importer_reference, importee_module, symbol)
                else:
                    definition = Definition(symbol, result.__dict__[symbol], importer_module)
                    symbol_definer = symbol_definers[definition]
                    if symbol_definer == importee_module:
                        if options.by_origin:
                            msg('Error: %s.%s is imported again into %s', importee_module, symbol, importer_reference)
                        else:
                            msg('Error: %s imports %s.%s again', importer_reference, importee_module, symbol)
                    elif options.by_origin:
                        msg('Error: %s.%s is imported by %s, which should import %s.%s instead', importee_module, symbol, importer_reference, symbol_definer, symbol)
                    else:
                        msg('Error: %s imports %s.%s but should import %s.%s instead', importer_reference, importee_module, symbol, symbol_definer, symbol)
    return result