class CoroStop(Exception):
    """Coroutine exit, as opposed to StopIteration which may
    mean it should be restarted."""
    pass