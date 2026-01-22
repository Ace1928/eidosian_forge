import contextlib
import io
import logging
import sys
import threading
import time
import unittest
from traits.api import HasTraits, Str, Int, Float, Any, Event
from traits.api import push_exception_handler, pop_exception_handler
class SlowListener(HasTraits):

    def handle_age_change(self):
        time.sleep(1.0)