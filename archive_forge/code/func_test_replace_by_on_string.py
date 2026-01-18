import yaql.tests
def test_replace_by_on_string(self):
    self.assertEqual('axxbyy', self.eval('a12b23.replaceBy(regex(`\\d+`), with(int($.value)) -> switch($ < 20 => xx, true => yy))'))
    self.assertEqual('axxb23', self.eval('a12b23.replaceBy(regex(`\\d+`), let(a => int($.value)) -> switch($a < 20 => xx, true => yy), 1)'))