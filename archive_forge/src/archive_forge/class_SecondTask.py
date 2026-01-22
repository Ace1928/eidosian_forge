import logging
import os
import sys
import time
import taskflow.engines
from taskflow import exceptions
from taskflow.patterns import unordered_flow as uf
from taskflow import task
from taskflow.tests import utils
from taskflow.types import failure
import example_utils as eu  # noqa
class SecondTask(task.Task):

    def execute(self, sleep2, raise2):
        time.sleep(sleep2)
        if not isinstance(raise2, bool):
            raise TypeError('Bad raise2 value: %r' % raise2)
        if raise2:
            raise SecondException('Second task failed')