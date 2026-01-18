import datetime
import logging
from cheroot.test import webtest
import pytest
import requests  # FIXME: Temporary using it directly, better switch
import cherrypy
from cherrypy.test.logtest import LogCase
def test_timez_log_format(log_tracker, monkeypatch, server):
    """Test a customized access_log_format string, which is a
    feature of _cplogging.LogManager.access()."""
    monkeypatch.setattr('cherrypy._cplogging.LogManager.access_log_format', '{h} {l} {u} {z} "{r}" {s} {b} "{f}" "{a}" {o}')
    log_tracker.markLog()
    expected_time = str(cherrypy._cplogging.LazyRfc3339UtcTime())
    monkeypatch.setattr('cherrypy._cplogging.LazyRfc3339UtcTime', lambda: expected_time)
    host = webtest.interface(webtest.WebCase.HOST)
    port = webtest.WebCase.PORT
    requests.get('http://%s:%s/as_string' % (host, port), headers={'Referer': 'REFERER', 'User-Agent': 'USERAGENT', 'Host': 'HOST'})
    log_tracker.assertLog(-1, '%s - - ' % host)
    log_tracker.assertLog(-1, expected_time)
    log_tracker.assertLog(-1, ' "GET /as_string HTTP/1.1" 200 7 "REFERER" "USERAGENT" HOST')