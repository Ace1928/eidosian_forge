import functools
import sys
import testtools
from testtools.matchers import Is
from fixtures import MonkeyPatch, TestWithFixtures
def test_patch_classmethod_with_function(self):
    oldmethod = C.foo_cls
    oldmethod_inst = C().foo_cls
    fixture = MonkeyPatch('fixtures.tests._fixtures.test_monkeypatch.C.foo_cls', fake)
    with fixture:
        cls, = C.foo_cls()
        self.expectThat(cls, Is(C))
        cls, arg = C().foo_cls(1)
        self.expectThat(cls, Is(C))
        self.assertEqual(1, arg)
    self._check_restored_static_or_class_method(oldmethod, oldmethod_inst, C, 'foo_cls')