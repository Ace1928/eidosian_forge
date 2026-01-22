import os
import sys
import threading
import time
import unittest.mock
import pytest
from cherrypy.process import wspbus
class Listner:
    """Bus handler return value tracker."""
    responses = []

    def get_listener(self, channel, index):
        """Return an argument tracking listener."""

        def listener(arg=None):
            self.responses.append(msg % (index, channel, arg))
        return listener