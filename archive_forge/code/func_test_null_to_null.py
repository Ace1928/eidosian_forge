import yaql.tests
def test_null_to_null(self):
    self.assertTrue(self.eval('null = null'))
    self.assertFalse(self.eval('null != null'))
    self.assertTrue(self.eval('null <= null'))
    self.assertTrue(self.eval('null >= null'))
    self.assertFalse(self.eval('null < null'))
    self.assertFalse(self.eval('null > null'))