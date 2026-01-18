import functools
import sys
import testtools
from testtools.matchers import Is
from fixtures import MonkeyPatch, TestWithFixtures
def test_patch_boundmethod_with_function(self):
    oldmethod = INST_C.foo
    fixture = MonkeyPatch('fixtures.tests._fixtures.test_monkeypatch.INST_C.foo', fake_no_args)
    with fixture:
        INST_C.foo()
    self.assertEqual(oldmethod, INST_C.foo)