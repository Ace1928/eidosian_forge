import fnmatch
import sys
import os
from inspect import CO_GENERATOR, CO_COROUTINE, CO_ASYNC_GENERATOR
def user_line(self, frame):
    import linecache
    name = frame.f_code.co_name
    if not name:
        name = '???'
    fn = self.canonic(frame.f_code.co_filename)
    line = linecache.getline(fn, frame.f_lineno, frame.f_globals)
    print('+++', fn, frame.f_lineno, name, ':', line.strip())