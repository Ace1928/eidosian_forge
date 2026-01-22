from __future__ import absolute_import, print_function, division
import abc
import logging
import sys
import time
from petl.compat import PY3
from petl.util.base import Table
from petl.util.statistics import onlinestats
class LoggingProgressView(ProgressViewBase):
    """
    Reports progress to a logger, log handler, or log adapter
    """

    def __init__(self, inner, batchsize, prefix, logger, level=logging.INFO):
        if logger is None:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(level)
        else:
            self.logger = logger
        self.level = level
        super(LoggingProgressView, self).__init__(inner, batchsize, prefix)

    def print_message(self, message):
        self.logger.log(self.level, message)