import yaql.tests
def test_replace_backref(self):
    self.assertEqual('Foo_Bar_Foo', self.eval('regex(`([a-z0-9])([A-Z])`).replace(FooBarFoo, `\\1_\\2`)'))