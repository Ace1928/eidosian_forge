from functools import wraps
def _embed_standard_shell(namespace={}, banner=''):
    """Start a standard python shell"""
    import code
    try:
        import readline
    except ImportError:
        pass
    else:
        import rlcompleter
        readline.parse_and_bind('tab:complete')

    @wraps(_embed_standard_shell)
    def wrapper(namespace=namespace, banner=''):
        code.interact(banner=banner, local=namespace)
    return wrapper