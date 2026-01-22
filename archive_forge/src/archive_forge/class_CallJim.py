import logging
import os
import sys
import taskflow.engines
from taskflow.patterns import linear_flow as lf
from taskflow import task
class CallJim(task.Task):

    def execute(self, jim_number, *args, **kwargs):
        print('Calling jim %s.' % jim_number)

    def revert(self, jim_number, *args, **kwargs):
        print('Calling %s and apologizing.' % jim_number)