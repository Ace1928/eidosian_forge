import functools
import logging
def allow_show_progress() -> bool:
    """Return False if any progressbar errors have occurred this session"""
    return _SHOW_PROGRESS