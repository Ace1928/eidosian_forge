import functools
import sys
import testtools
from testtools.matchers import Is
from fixtures import MonkeyPatch, TestWithFixtures
def test_patch_missing_attribute(self):
    fixture = MonkeyPatch('fixtures.tests._fixtures.test_monkeypatch.new_attr', True)
    self.assertFalse('new_attr' in globals())
    fixture.setUp()
    try:
        self.assertEqual(True, new_attr)
    finally:
        fixture.cleanUp()
        self.assertFalse('new_attr' in globals())