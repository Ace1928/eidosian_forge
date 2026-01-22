import inspect
import random
import re
import unittest
from . import import_submodule
class PygameTestLoader(unittest.TestLoader):

    def __init__(self, randomize_tests=False, include_incomplete=False, exclude=('interactive',)):
        super().__init__()
        self.randomize_tests = randomize_tests
        if exclude is None:
            self.exclude = set()
        else:
            self.exclude = set(exclude)
        if include_incomplete:
            self.testMethodPrefix = ('test', 'todo_')

    def getTestCaseNames(self, testCaseClass):
        res = []
        for name in super().getTestCaseNames(testCaseClass):
            tags = get_tags(testCaseClass, getattr(testCaseClass, name))
            if self.exclude.isdisjoint(tags):
                res.append(name)
        if self.randomize_tests:
            random.shuffle(res)
        return res