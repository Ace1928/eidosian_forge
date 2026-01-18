import funcsigs
import unittest2 as unittest
def test_user_type(self):

    class dummy(object):
        pass
    self.assertEqual(funcsigs.formatannotation(dummy), 'tests.test_formatannotation.dummy')