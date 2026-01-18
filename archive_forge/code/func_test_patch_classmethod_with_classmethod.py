import functools
import sys
import testtools
from testtools.matchers import Is
from fixtures import MonkeyPatch, TestWithFixtures
def test_patch_classmethod_with_classmethod(self):
    oldmethod = C.foo_cls
    oldmethod_inst = C().foo_cls
    fixture = MonkeyPatch('fixtures.tests._fixtures.test_monkeypatch.C.foo_cls', D.bar_cls_args)
    with fixture:
        if NEW_PY39_CLASSMETHOD:
            cls, = C.foo_cls()
            self.expectThat(cls, Is(D))
            cls, = C().foo_cls()
            self.expectThat(cls, Is(D))
        else:
            cls, target_class = C.foo_cls()
            self.expectThat(cls, Is(D))
            self.expectThat(target_class, Is(C))
            cls, target_class = C().foo_cls()
            self.expectThat(cls, Is(D))
            self.expectThat(target_class, Is(C))
    self._check_restored_static_or_class_method(oldmethod, oldmethod_inst, C, 'foo_cls')