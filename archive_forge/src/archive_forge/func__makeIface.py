import unittest
def _makeIface():
    from zope.interface import Interface

    class IDummy(Interface):
        pass
    return IDummy