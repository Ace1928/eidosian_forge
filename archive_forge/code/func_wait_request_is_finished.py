import threading
from base64 import b64encode
from datetime import datetime
from time import sleep
import certifi
import pytest
import responses
from kivy.network.urlrequest import UrlRequestRequests as UrlRequest
from requests.auth import HTTPBasicAuth
from responses import matchers
def wait_request_is_finished(self, kivy_clock, request, timeout=10):
    start_time = datetime.now()
    timed_out = False
    while not request.is_finished and (not timed_out):
        kivy_clock.tick()
        sleep(0.1)
        timed_out = (datetime.now() - start_time).total_seconds() > timeout
    assert request.is_finished