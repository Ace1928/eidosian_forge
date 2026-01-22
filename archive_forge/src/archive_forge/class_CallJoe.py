import logging
import os
import sys
import taskflow.engines
from taskflow.patterns import linear_flow as lf
from taskflow import task
class CallJoe(task.Task):

    def execute(self, joe_number, *args, **kwargs):
        print('Calling joe %s.' % joe_number)

    def revert(self, joe_number, *args, **kwargs):
        print('Calling %s and apologizing.' % joe_number)