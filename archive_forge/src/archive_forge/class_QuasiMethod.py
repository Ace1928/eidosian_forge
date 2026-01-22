import unittest
class QuasiMethod(Method):

    def __call__(self, *args, **kw):
        raise NotImplementedError()