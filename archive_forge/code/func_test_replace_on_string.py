import yaql.tests
def test_replace_on_string(self):
    self.assertEqual('axxbxx', self.eval('a12b23.replace(regex(`\\d+`), xx)'))
    self.assertEqual('axxb23', self.eval('a12b23.replace(regex(`\\d+`), xx, 1)'))