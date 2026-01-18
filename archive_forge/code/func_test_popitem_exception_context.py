import unittest
def test_popitem_exception_context(self):
    exception = None
    try:
        self.Cache(maxsize=2).popitem()
    except Exception as e:
        exception = e
    self.assertIsNone(exception.__cause__)
    self.assertTrue(exception.__suppress_context__)