from unittest import TestCase
import macaroonbakery.httpbakery as httpbakery
import requests
from mock import patch
from httmock import HTTMock, response, urlmatch
@urlmatch(path='.*/wait')
def wait_after_401(url, request):
    if request.url != 'http://example.com/wait':
        return {'status_code': 500}
    return {'status_code': 200, 'content': {'DischargeToken': discharge_token, 'Macaroon': discharged_macaroon}}