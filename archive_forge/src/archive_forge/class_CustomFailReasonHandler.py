from hashlib import md5
import unittest
from tornado.escape import utf8
from tornado.testing import AsyncHTTPTestCase
from tornado.test import httpclient_test
from tornado.web import Application, RequestHandler
class CustomFailReasonHandler(RequestHandler):

    def get(self):
        self.set_status(400, 'Custom reason')