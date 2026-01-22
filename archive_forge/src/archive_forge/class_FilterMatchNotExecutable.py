import configparser
import logging
import logging.handlers
import os
import signal
import sys
from oslo_rootwrap import filters
from oslo_rootwrap import subprocess
class FilterMatchNotExecutable(Exception):
    """Raised when a filter matched but no executable was found."""

    def __init__(self, match=None, **kwargs):
        self.match = match