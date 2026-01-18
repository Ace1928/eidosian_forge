import functools
import sys
import testtools
from testtools.matchers import Is
from fixtures import MonkeyPatch, TestWithFixtures
def test_patch_function_with_partial(self):
    oldmethod = fake_no_args
    fixture = MonkeyPatch('fixtures.tests._fixtures.test_monkeypatch.fake_no_args', functools.partial(fake, 1))
    with fixture:
        ret, = fake_no_args()
        self.assertEqual(1, ret)
    self.assertEqual(oldmethod, fake_no_args)