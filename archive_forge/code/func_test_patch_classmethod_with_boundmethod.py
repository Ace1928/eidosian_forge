import functools
import sys
import testtools
from testtools.matchers import Is
from fixtures import MonkeyPatch, TestWithFixtures
def test_patch_classmethod_with_boundmethod(self):
    oldmethod = C.foo_cls
    oldmethod_inst = C().foo_cls
    d = D()
    fixture = MonkeyPatch('fixtures.tests._fixtures.test_monkeypatch.C.foo_cls', d.bar_two_args)
    with fixture:
        slf, cls = C.foo_cls()
        self.expectThat(slf, Is(d))
        if NEW_PY39_CLASSMETHOD:
            self.expectThat(cls, Is(None))
        else:
            self.expectThat(cls, Is(C))
        slf, cls = C().foo_cls()
        self.expectThat(slf, Is(d))
        if NEW_PY39_CLASSMETHOD:
            self.expectThat(cls, Is(None))
        else:
            self.expectThat(cls, Is(C))
    self._check_restored_static_or_class_method(oldmethod, oldmethod_inst, C, 'foo_cls')