import sys
import unittest
from webtest import TestApp
from pecan import expose, make_app
from pecan.secure import secure, unlocked, SecureController
from pecan.tests import PecanTestCase
class DeepSecretController(SecureController):
    authorized = False

    @expose()
    @unlocked
    def _lookup(self, someID, *remainder):
        if someID == 'notfound':
            return None
        return (SubController(someID), remainder)

    @expose()
    def index(self):
        return 'Deep Secret'

    @classmethod
    def check_permissions(cls):
        permissions_checked.add('deepsecret')
        return cls.authorized