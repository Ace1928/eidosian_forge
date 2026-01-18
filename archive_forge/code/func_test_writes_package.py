import os.path
import testtools
from fixtures import PythonPackage, TestWithFixtures
def test_writes_package(self):
    fixture = PythonPackage('foo', [('bar.py', b'woo')])
    fixture.setUp()
    try:
        self.assertEqual('', open(os.path.join(fixture.base, 'foo', '__init__.py')).read())
        self.assertEqual('woo', open(os.path.join(fixture.base, 'foo', 'bar.py')).read())
    finally:
        fixture.cleanUp()