import yaql.tests
def test_search_all(self):
    self.assertEqual(['24', '16'], self.eval("regex(`\\d+`).searchAll('a24.16b')"))