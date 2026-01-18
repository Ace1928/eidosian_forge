import functools
import sys
import testtools
from testtools.matchers import Is
from fixtures import MonkeyPatch, TestWithFixtures
def test_patch_unboundmethod_with_function(self):
    oldmethod = C.foo
    fixture = MonkeyPatch('fixtures.tests._fixtures.test_monkeypatch.C.foo', fake)
    with fixture:
        c = C()
        target_self, arg = c.foo(1)
        self.expectThat(target_self, Is(c))
        self.assertTrue(1, arg)
    self.assertEqual(oldmethod, C.foo)