from unittest import mock
from oslo_versionedobjects import exception
from oslo_versionedobjects import test
def test_vo_exception(self):
    exc = exception.VersionedObjectsException()
    self.assertEqual('An unknown exception occurred.', str(exc))
    self.assertEqual({'code': 500}, exc.kwargs)