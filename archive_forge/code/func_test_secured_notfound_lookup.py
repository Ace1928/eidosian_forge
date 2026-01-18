import sys
import unittest
from webtest import TestApp
from pecan import expose, make_app
from pecan.secure import secure, unlocked, SecureController
from pecan.tests import PecanTestCase
def test_secured_notfound_lookup(self):
    response = self.app.get('/secret/notfound/', expect_errors=True)
    assert response.status_int == 404