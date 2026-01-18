import yaql.tests
def test_shift_bits_right(self):
    self.assertEqual(2, self.eval('shiftBitsRight(32, 4)'))
    self.assertEqual(0, self.eval('shiftBitsRight(32, 6)'))