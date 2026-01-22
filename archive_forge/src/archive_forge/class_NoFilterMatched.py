import configparser
import logging
import logging.handlers
import os
import signal
import sys
from oslo_rootwrap import filters
from oslo_rootwrap import subprocess
class NoFilterMatched(Exception):
    """This exception is raised when no filter matched."""
    pass