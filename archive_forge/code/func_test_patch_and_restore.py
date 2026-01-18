import functools
import sys
import testtools
from testtools.matchers import Is
from fixtures import MonkeyPatch, TestWithFixtures
def test_patch_and_restore(self):
    fixture = MonkeyPatch('fixtures.tests._fixtures.test_monkeypatch.reference', 45)
    self.assertEqual(23, reference)
    fixture.setUp()
    try:
        self.assertEqual(45, reference)
    finally:
        fixture.cleanUp()
        self.assertEqual(23, reference)