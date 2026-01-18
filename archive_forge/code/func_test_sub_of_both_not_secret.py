import sys
import unittest
from webtest import TestApp
from pecan import expose, make_app
from pecan.secure import secure, unlocked, SecureController
from pecan.tests import PecanTestCase
def test_sub_of_both_not_secret(self):
    response = self.app.get('/notsecret/hi/')
    assert response.status_int == 200
    assert response.body == b'Index hi'