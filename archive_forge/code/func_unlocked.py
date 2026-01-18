import sys
import unittest
from webtest import TestApp
from pecan import expose, make_app
from pecan.secure import secure, unlocked, SecureController
from pecan.tests import PecanTestCase
@expose()
@secure(lambda: True)
def unlocked(self):
    return 'Sure thing'