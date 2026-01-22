from __future__ import print_function
import sys
import os
import types
import traceback
from abc import abstractmethod
class ObjectCounter(object):

    def __init__(self):
        self.map = {}

    def __call__(self, k):
        self.map[k] = self.map.get(k, 0) + 1

    def dump(self):
        for k in sorted(self.map):
            sys.stdout.write('{} -> {}'.format(k, self.map[k]))