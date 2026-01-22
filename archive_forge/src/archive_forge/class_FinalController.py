from json import dumps, loads
import unittest
import struct
import sys
import warnings
from webtest import TestApp
from pecan import abort, expose, make_app, response, redirect
from pecan.rest import RestController
from pecan.tests import PecanTestCase
class FinalController(RestController):

    def __init__(self, id_):
        self.id_ = id_

    @expose()
    def get_all(self):
        return 'FINAL-%s' % self.id_

    @expose()
    def post(self):
        return 'POST-%s' % self.id_