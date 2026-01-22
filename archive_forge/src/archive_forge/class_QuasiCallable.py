import unittest
class QuasiCallable:

    def __call__(self, *args, **kw):
        raise NotImplementedError()