import string
from taskflow import test
from taskflow.utils import iter_utils
def test_while_is_not(self):

    class Dummy(object):

        def __init__(self, char):
            self.char = char
    dummy_list = [Dummy(a) for a in string.ascii_lowercase]
    it = iter(dummy_list)
    self.assertEqual([dummy_list[0]], list(iter_utils.while_is_not(it, dummy_list[0])))
    it = iter(dummy_list)
    self.assertEqual(dummy_list[0:2], list(iter_utils.while_is_not(it, dummy_list[1])))
    self.assertEqual(dummy_list[2:], list(iter_utils.while_is_not(it, Dummy('zzz'))))
    it = iter(dummy_list)
    self.assertEqual(dummy_list, list(iter_utils.while_is_not(it, Dummy(''))))