from unittest import TestCase
import macaroonbakery.httpbakery as httpbakery
import requests
from mock import patch
from httmock import HTTMock, response, urlmatch
@urlmatch(path='.*/visit')
def visit_200(url, request):
    return {'status_code': 200, 'content': {'interactive': '/visit'}}