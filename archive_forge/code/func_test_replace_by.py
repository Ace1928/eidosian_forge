import yaql.tests
def test_replace_by(self):
    self.assertEqual('axxbyy', self.eval('regex(`\\d+`).replaceBy(a12b23, let(a => int($.value)) -> switch($a < 20 => xx, true => yy))'))
    self.assertEqual('axxb23', self.eval('regex(`\\d+`).replaceBy(a12b23, let(a => int($.value)) -> switch($a < 20 => xx, true => yy), 1)'))