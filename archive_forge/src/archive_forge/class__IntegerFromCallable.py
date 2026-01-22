from __future__ import unicode_literals
from abc import ABCMeta, abstractmethod
from six import with_metaclass
class _IntegerFromCallable(Integer):

    def __init__(self, func=0):
        self.func = func

    def __repr__(self):
        return 'Integer.from_callable(%r)' % self.func

    def __int__(self):
        return int(self.func())