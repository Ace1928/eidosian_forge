import contextlib
import functools
import threading
import time
from unittest import mock
import eventlet
from eventlet.green import threading as green_threading
import testscenarios
import futurist
from futurist import periodics
from futurist.tests import base
def trap_failures(cb, kind, periodic_spacing, exc_info, traceback=None):
    captures.append([cb, kind, periodic_spacing, traceback])
    ev.set()