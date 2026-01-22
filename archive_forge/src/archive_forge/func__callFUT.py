import unittest
def _callFUT(self, iface, klass, **kwargs):
    return self.verifier(iface, self._adjust_object_before_verify(klass), **kwargs)