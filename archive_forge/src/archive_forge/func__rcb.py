from __future__ import annotations
from kombu.utils.eventio import ERR, READ, WRITE
from kombu.utils.functional import reprcall
def _rcb(obj):
    if obj is None:
        return '<missing>'
    if isinstance(obj, str):
        return obj
    if isinstance(obj, tuple):
        cb, args = obj
        return reprcall(cb.__name__, args=args)
    return obj.__name__