import logging
import os
import sys
from taskflow import engines
from taskflow.patterns import linear_flow
from taskflow.patterns import unordered_flow
from taskflow import task
class TotalReducer(task.Task):

    def execute(self, *args, **kwargs):
        total = 0
        for k, v in kwargs.items():
            if k.startswith('reduction_'):
                total += v
        return total