import logging
import os
import sys
from taskflow import engines
from taskflow.patterns import linear_flow
from taskflow import task
class TaskA(task.Task):
    default_provides = 'a'

    def execute(self):
        print("Executing '%s'" % self.name)
        return 'a'