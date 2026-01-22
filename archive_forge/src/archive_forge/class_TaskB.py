import logging
import os
import sys
from taskflow import engines
from taskflow.patterns import linear_flow
from taskflow import task
class TaskB(task.Task):

    def execute(self, a):
        print("Executing '%s'" % self.name)
        print("Got input '%s'" % a)