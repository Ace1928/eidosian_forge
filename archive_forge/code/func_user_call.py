import fnmatch
import sys
import os
from inspect import CO_GENERATOR, CO_COROUTINE, CO_ASYNC_GENERATOR
def user_call(self, frame, args):
    name = frame.f_code.co_name
    if not name:
        name = '???'
    print('+++ call', name, args)