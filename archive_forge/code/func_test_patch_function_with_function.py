import functools
import sys
import testtools
from testtools.matchers import Is
from fixtures import MonkeyPatch, TestWithFixtures
def test_patch_function_with_function(self):
    oldmethod = fake_no_args
    fixture = MonkeyPatch('fixtures.tests._fixtures.test_monkeypatch.fake_no_args', fake_no_args2)
    with fixture:
        fake_no_args()
    self.assertEqual(oldmethod, fake_no_args)