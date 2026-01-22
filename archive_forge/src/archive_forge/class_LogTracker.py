import os
import sys
import threading
import time
import unittest.mock
import pytest
from cherrypy.process import wspbus
class LogTracker:
    """Bus log tracker."""
    log_entries = []

    def __init__(self, bus):

        def logit(msg, level):
            self.log_entries.append(msg)
        bus.subscribe('log', logit)