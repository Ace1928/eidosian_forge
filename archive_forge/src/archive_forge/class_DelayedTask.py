import logging
import os
import random
import sys
import time
import futurist
from taskflow import engines
from taskflow.patterns import unordered_flow as uf
from taskflow import task
from taskflow.utils import threading_utils as tu
class DelayedTask(task.Task):

    def __init__(self, name):
        super(DelayedTask, self).__init__(name=name)
        self._wait_for = random.random()

    def execute(self):
        print("Running '%s' in thread '%s'" % (self.name, tu.get_ident()))
        time.sleep(self._wait_for)