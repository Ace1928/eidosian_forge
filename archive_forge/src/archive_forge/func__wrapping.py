from concurrent import futures
import logging
def _wrapping(*args, **kwargs):
    try:
        return behavior(*args, **kwargs)
    except Exception:
        _LOGGER.exception('Unexpected exception from %s executed in logging pool!', behavior)
        raise