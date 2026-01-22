import logging
import os
import sys
import tempfile
import traceback
from taskflow import engines
from taskflow.patterns import linear_flow as lf
from taskflow.persistence import models
from taskflow import task
import example_utils as eu  # noqa
class ByeTask(task.Task):

    def __init__(self, blowup):
        super(ByeTask, self).__init__()
        self._blowup = blowup

    def execute(self):
        if self._blowup:
            raise Exception('Fail!')
        print('Bye!')