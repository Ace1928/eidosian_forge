import functools
import sys
import testtools
from testtools.matchers import Is
from fixtures import MonkeyPatch, TestWithFixtures
def _check_restored_static_or_class_method(self, oldmethod, oldmethod_inst, klass, methodname):
    restored_method = getattr(klass, methodname)
    restored_method_inst = getattr(klass(), methodname)
    self.assertEqual(oldmethod, restored_method)
    self.assertEqual(oldmethod, restored_method_inst)
    self.assertEqual(oldmethod_inst, restored_method)
    self.assertEqual(oldmethod_inst, restored_method_inst)
    restored_method()
    restored_method_inst()