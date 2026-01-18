from .. import tests
from . import features
def test_not_lines(self):
    self.assertRaises(TypeError, self.module.chunks_to_lines, object())
    self.assertRaises(TypeError, self.module.chunks_to_lines, [object()])
    self.assertRaises(TypeError, self.module.chunks_to_lines, [b'foo', object()])