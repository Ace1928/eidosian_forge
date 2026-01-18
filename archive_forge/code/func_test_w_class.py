import sys
import unittest
import sys
def test_w_class(self):
    from zope.interface.tests import advisory_testing
    kind, module, f_locals, f_globals = advisory_testing.NewStyleClass.classLevelFrameInfo
    self.assertEqual(kind, 'class')
    for d in (module.__dict__, f_globals):
        self.assertTrue(d is advisory_testing.my_globals)