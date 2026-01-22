import logging
import os
import sys
import taskflow.engines
from taskflow.patterns import linear_flow as lf
from taskflow import task
class Multiplier(task.Task):

    def __init__(self, name, multiplier, provides=None, rebind=None):
        super(Multiplier, self).__init__(name=name, provides=provides, rebind=rebind)
        self._multiplier = multiplier

    def execute(self, z):
        return z * self._multiplier