import inspect
import os
import threading
from contextlib import contextmanager
from datetime import datetime
from traits import trait_notifiers
class ChangeMessageRecord(object):
    """ Message record for a change event dispatch.

    """
    __slots__ = ('time', 'indent', 'name', 'old', 'new', 'class_name')

    def __init__(self, time, indent, name, old, new, class_name):
        self.time = time
        self.indent = indent
        self.name = name
        self.old = old
        self.new = new
        self.class_name = class_name

    def __str__(self):
        length = self.indent * 2
        return CHANGEMSG.format(time=self.time, direction='>', name=self.name, old=self.old, new=self.new, class_name=self.class_name, length=length)