import unittest
def test_w_callable_non_func_method(self):
    from zope.interface import Interface
    from zope.interface import implementer
    from zope.interface.interface import Method

    class QuasiMethod(Method):

        def __call__(self, *args, **kw):
            raise NotImplementedError()

    class QuasiCallable:

        def __call__(self, *args, **kw):
            raise NotImplementedError()

    class ICurrent(Interface):
        attr = QuasiMethod('This is callable')

    @implementer(ICurrent)
    class Current:
        attr = QuasiCallable()
    self._callFUT(ICurrent, Current)