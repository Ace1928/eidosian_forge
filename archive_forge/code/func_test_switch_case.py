import yaql.tests
def test_switch_case(self):
    expr = "$.switchCase('a', 'b', 'c')"
    self.assertEqual('a', self.eval(expr, data=0))
    self.assertEqual('b', self.eval(expr, data=1))
    self.assertEqual('c', self.eval(expr, data=3))
    self.assertEqual('c', self.eval(expr, data=30))
    self.assertEqual('c', self.eval(expr, data=-30))