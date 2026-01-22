import sys
import datetime
import locale as _locale
from itertools import repeat
class _localized_day:
    _days = [datetime.date(2001, 1, i + 1).strftime for i in range(7)]

    def __init__(self, format):
        self.format = format

    def __getitem__(self, i):
        funcs = self._days[i]
        if isinstance(i, slice):
            return [f(self.format) for f in funcs]
        else:
            return funcs(self.format)

    def __len__(self):
        return 7