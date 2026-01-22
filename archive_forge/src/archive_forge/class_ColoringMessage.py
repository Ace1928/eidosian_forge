import re
from string import Formatter
class ColoringMessage(str):
    __fields__ = ('_messages',)

    def __format__(self, spec):
        return next(self._messages).__format__(spec)