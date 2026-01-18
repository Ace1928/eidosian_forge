import unittest
from twisted.internet import defer
def test_unhandledDeferred(self):
    try:
        1 / 0
    except ZeroDivisionError:
        f = defer.fail()
    l = [f]
    l.append(l)