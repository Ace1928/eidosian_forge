import yaql.tests
def test_not_matches_operator_regex(self):
    self.assertFalse(self.eval("axb !~ regex('a.b')"))
    self.assertTrue(self.eval("abx !~ regex('a.b')"))