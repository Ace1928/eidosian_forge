from __future__ import annotations
import logging
from vine.utils import wraps
from kombu.log import get_logger
@wraps(meth)
def __wrapped(*args, **kwargs):
    info = ''
    if self.ident:
        info += self.ident.format(self.instance)
    info += f'{meth.__name__}('
    if args:
        info += ', '.join(map(repr, args))
    if kwargs:
        if args:
            info += ', '
        info += ', '.join((f'{key}={value!r}' for key, value in kwargs.items()))
    info += ')'
    self.logger.debug(info)
    return meth(*args, **kwargs)