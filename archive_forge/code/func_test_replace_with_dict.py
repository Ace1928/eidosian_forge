from yaql.language import exceptions
import yaql.tests
def test_replace_with_dict(self):
    self.assertEqual('Az1D', self.eval('AxyD.replace({x => z, y => 1})'))
    self.assertEqual('Ayfalse2D!', self.eval("A122Dnull.replace({1 => y, 2 => false, null => '!'}, 1)"))