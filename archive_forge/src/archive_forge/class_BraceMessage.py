import logging
from string import Template
class BraceMessage(object):

    def __init__(self, fmt, *args, **kwargs):
        self.fmt = fmt
        self.args = args
        self.kwargs = kwargs
        self.str = None

    def __str__(self):
        if self.str is None:
            self.str = self.fmt.format(*self.args, **self.kwargs)
        return self.str