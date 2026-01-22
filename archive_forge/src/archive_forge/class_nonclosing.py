import logging
import sys
import threading
from oslo_rootwrap import cmd
from oslo_rootwrap import subprocess
class nonclosing(object):

    def __init__(self, f):
        self._f = f

    def __getattr__(self, name):
        return getattr(self._f, name)

    def close(self):
        pass