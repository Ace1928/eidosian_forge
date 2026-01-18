import datetime
import logging
from cheroot.test import webtest
import pytest
import requests  # FIXME: Temporary using it directly, better switch
import cherrypy
from cherrypy.test.logtest import LogCase
def test_normal_yield(log_tracker, server):
    log_tracker.markLog()
    host = webtest.interface(webtest.WebCase.HOST)
    port = webtest.WebCase.PORT
    resp = requests.get('http://%s:%s/as_yield' % (host, port), headers={'User-Agent': ''})
    expected_body = 'content'
    assert resp.text == expected_body
    assert resp.status_code == 200
    intro = '%s - - [' % host
    log_tracker.assertLog(-1, intro)
    content_length = len(expected_body)
    if not any((k for k, v in resp.headers.items() if k.lower() == 'content-length')):
        content_length = '-'
    log_tracker.assertLog(-1, '] "GET /as_yield HTTP/1.1" 200 %s "" ""' % content_length)