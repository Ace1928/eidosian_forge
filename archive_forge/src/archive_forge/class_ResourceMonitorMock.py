import os
import numpy as np
import threading
from time import time
from .. import config, logging
class ResourceMonitorMock:
    """A mock class to use when the monitor is disabled."""

    @property
    def fname(self):
        """Get/set the internal filename"""
        return None

    def __init__(self, pid, freq=5, fname=None, python=True):
        pass

    def start(self):
        pass

    def stop(self):
        return {}