from unittest import TestCase
import macaroonbakery.httpbakery as httpbakery
import requests
from mock import patch
from httmock import HTTMock, response, urlmatch
@urlmatch(path='.*/someprotecteurl')
def valid_200(url, request):
    return {'status_code': 200, 'content': {'Value': 'some value'}}