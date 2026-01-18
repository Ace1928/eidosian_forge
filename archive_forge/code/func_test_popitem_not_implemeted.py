from .. import fifo_cache, tests
def test_popitem_not_implemeted(self):
    c = fifo_cache.FIFOCache()
    self.assertRaises(NotImplementedError, c.popitem)