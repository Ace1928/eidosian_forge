from twisted.python import usage
from twisted.trial import unittest
class WeirdCallableOptions(usage.Options):

    def _bar(value):
        raise RuntimeError('Ouch')

    def _foo(value):
        raise ValueError('Yay')
    optParameters = [['barwrong', None, None, 'Bar with strange callable', _bar], ['foowrong', None, None, 'Foo with strange callable', _foo]]