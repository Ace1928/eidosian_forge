import dill
import abc
from abc import ABC
import warnings
from types import FunctionType
class EasyAsAbc(OneTwoThree):

    def __init__(self):
        self._bar = None

    def foo(self):
        return 'Instance Method FOO'

    @property
    def bar(self):
        return self._bar

    @bar.setter
    def bar(self, value):
        self._bar = value

    @classmethod
    def cfoo(cls):
        return 'Class Method CFOO'

    @staticmethod
    def sfoo():
        return 'Static Method SFOO'