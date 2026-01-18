import unittest
def test_w_decorated_method(self):
    from zope.interface import Interface
    from zope.interface import implementer

    def decorator(func):
        return property(lambda self: func.__get__(self))

    class ICurrent(Interface):

        def method(a):
            """docstring"""

    @implementer(ICurrent)
    class Current:

        @decorator
        def method(self, a):
            raise NotImplementedError()
    self._callFUT(ICurrent, Current)