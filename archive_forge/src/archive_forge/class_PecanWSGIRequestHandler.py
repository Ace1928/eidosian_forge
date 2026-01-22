from __future__ import print_function
import logging
import os
import sys
import threading
import time
import subprocess
from wsgiref.simple_server import WSGIRequestHandler
from pecan.commands import BaseCommand
from pecan import util
class PecanWSGIRequestHandler(WSGIRequestHandler, object):
    """
    A wsgiref request handler class that allows actual log output depending on
    the application configuration.
    """

    def __init__(self, *args, **kwargs):
        self.path = ''
        super(PecanWSGIRequestHandler, self).__init__(*args, **kwargs)

    def log_message(self, format, *args):
        """
        overrides the ``log_message`` method from the wsgiref server so that
        normal logging works with whatever configuration the application has
        been set to.

        Levels are inferred from the HTTP status code, 4XX codes are treated as
        warnings, 5XX as errors and everything else as INFO level.
        """
        code = args[1][0]
        levels = {'4': 'warning', '5': 'error'}
        log_handler = getattr(logger, levels.get(code, 'info'))
        log_handler(format % args)