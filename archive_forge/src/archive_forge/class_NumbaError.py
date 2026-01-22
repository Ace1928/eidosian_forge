import abc
import contextlib
import os
import sys
import warnings
import numba.core.config
import numpy as np
from collections import defaultdict
from functools import wraps
from abc import abstractmethod
class NumbaError(Exception):

    def __init__(self, msg, loc=None, highlighting=True):
        self.msg = msg
        self.loc = loc
        if highlighting:
            highlight = termcolor().errmsg
        else:

            def highlight(x):
                return x
        if loc:
            new_msg = '%s\n%s\n' % (msg, loc.strformat())
        else:
            new_msg = '%s' % (msg,)
        super(NumbaError, self).__init__(highlight(new_msg))

    @property
    def contexts(self):
        try:
            return self._contexts
        except AttributeError:
            self._contexts = lst = []
            return lst

    def add_context(self, msg):
        """
        Add contextual info.  The exception message is expanded with the new
        contextual information.
        """
        self.contexts.append(msg)
        f = termcolor().errmsg('{0}\n') + termcolor().filename('During: {1}')
        newmsg = f.format(self, msg)
        self.args = (newmsg,)
        return self

    def patch_message(self, new_message):
        """
        Change the error message to the given new message.
        """
        self.args = (new_message,) + self.args[1:]