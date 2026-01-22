from json import dumps, loads
import unittest
import struct
import sys
import warnings
from webtest import TestApp
from pecan import abort, expose, make_app, response, redirect
from pecan.rest import RestController
from pecan.tests import PecanTestCase
class BarController(RestController):

    @expose()
    def get_all(self):
        return 'BAR'

    @expose()
    def put(self):
        return 'PUT_BAR'

    @expose()
    def delete(self):
        return 'DELETE_BAR'