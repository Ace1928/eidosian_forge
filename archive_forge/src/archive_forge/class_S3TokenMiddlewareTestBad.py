from unittest import mock
import urllib.parse
import fixtures
from oslo_serialization import jsonutils
import requests
from requests_mock.contrib import fixture as rm_fixture
from testtools import matchers
import webob
from keystonemiddleware import s3_token
from keystonemiddleware.tests.unit import utils
class S3TokenMiddlewareTestBad(S3TokenMiddlewareTestBase):

    def setUp(self):
        super(S3TokenMiddlewareTestBad, self).setUp()
        self.middleware = s3_token.S3Token(FakeApp(), self.conf)

    def test_unauthorized_token(self):
        ret = {'error': {'message': 'EC2 access key not found.', 'code': 401, 'title': 'Unauthorized'}}
        self.requests_mock.post(self.TEST_URL, status_code=403, json=ret)
        req = webob.Request.blank('/v1/AUTH_cfa/c/o')
        req.headers['Authorization'] = 'access:signature'
        req.headers['X-Storage-Token'] = 'token'
        resp = req.get_response(self.middleware)
        s3_denied_req = self.middleware._deny_request('AccessDenied')
        self.assertEqual(resp.body, s3_denied_req.body)
        self.assertEqual(resp.status_int, s3_denied_req.status_int)

    def test_bogus_authorization(self):
        req = webob.Request.blank('/v1/AUTH_cfa/c/o')
        req.headers['Authorization'] = 'badboy'
        req.headers['X-Storage-Token'] = 'token'
        resp = req.get_response(self.middleware)
        self.assertEqual(resp.status_int, 400)
        s3_invalid_req = self.middleware._deny_request('InvalidURI')
        self.assertEqual(resp.body, s3_invalid_req.body)
        self.assertEqual(resp.status_int, s3_invalid_req.status_int)

    def test_fail_to_connect_to_keystone(self):
        with mock.patch.object(self.middleware, '_json_request') as o:
            s3_invalid_req = self.middleware._deny_request('InvalidURI')
            o.side_effect = s3_token.ServiceError(s3_invalid_req)
            req = webob.Request.blank('/v1/AUTH_cfa/c/o')
            req.headers['Authorization'] = 'access:signature'
            req.headers['X-Storage-Token'] = 'token'
            resp = req.get_response(self.middleware)
            self.assertEqual(resp.body, s3_invalid_req.body)
            self.assertEqual(resp.status_int, s3_invalid_req.status_int)

    def test_bad_reply(self):
        self.requests_mock.post(self.TEST_URL, status_code=201, text='<badreply>')
        req = webob.Request.blank('/v1/AUTH_cfa/c/o')
        req.headers['Authorization'] = 'access:signature'
        req.headers['X-Storage-Token'] = 'token'
        resp = req.get_response(self.middleware)
        s3_invalid_req = self.middleware._deny_request('InvalidURI')
        self.assertEqual(resp.body, s3_invalid_req.body)
        self.assertEqual(resp.status_int, s3_invalid_req.status_int)