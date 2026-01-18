import sys
import unittest
from webtest import TestApp
from pecan import expose, make_app
from pecan.secure import secure, unlocked, SecureController
from pecan.tests import PecanTestCase
def test_wrapped_attribute_success(self):
    self.secret_cls.independent_authorization = True
    response = self.app.get('/secret/wrapped/')
    assert response.status_int == 200
    assert response.body == b'Index wrapped'
    assert len(self.permissions_checked) == 1
    assert 'independent' in self.permissions_checked