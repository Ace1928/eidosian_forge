import math as _math
import numbers as _numbers
import sys
import contextvars
import re
class _ContextManager(object):
    """Context manager class to support localcontext().

      Sets a copy of the supplied context in __enter__() and restores
      the previous decimal context in __exit__()
    """

    def __init__(self, new_context):
        self.new_context = new_context.copy()

    def __enter__(self):
        self.saved_context = getcontext()
        setcontext(self.new_context)
        return self.new_context

    def __exit__(self, t, v, tb):
        setcontext(self.saved_context)