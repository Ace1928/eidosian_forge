import yaql.tests
def test_select_case(self):
    expr = 'selectCase($ < 10, $ >= 10 and $ < 100)'
    self.assertEqual(2, self.eval(expr, data=123))
    self.assertEqual(1, self.eval(expr, data=50))
    self.assertEqual(0, self.eval(expr, data=-123))