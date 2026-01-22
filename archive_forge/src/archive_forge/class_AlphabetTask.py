import fractions
import functools
import logging
import os
import string
import sys
import time
from taskflow import engines
from taskflow import exceptions
from taskflow.patterns import linear_flow
from taskflow import task
class AlphabetTask(task.Task):
    _DELAY = 0.1
    _PROGRESS_PARTS = [fractions.Fraction('%s/5' % x) for x in range(1, 5)]

    def execute(self):
        for p in self._PROGRESS_PARTS:
            self.update_progress(p)
            time.sleep(self._DELAY)