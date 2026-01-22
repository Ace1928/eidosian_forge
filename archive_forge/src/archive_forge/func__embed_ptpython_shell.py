from functools import wraps
def _embed_ptpython_shell(namespace={}, banner=''):
    """Start a ptpython shell"""
    import ptpython.repl

    @wraps(_embed_ptpython_shell)
    def wrapper(namespace=namespace, banner=''):
        print(banner)
        ptpython.repl.embed(locals=namespace)
    return wrapper