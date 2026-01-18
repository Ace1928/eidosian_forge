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
@responses.activate
def test_on_finish(self, kivy_clock):
    _queue = UrlRequestQueue([])
    responses.get(self.url, body='{}', status=400, content_type='application/json')
    req = UrlRequest(self.url, on_finish=_queue._on_finish, debug=True)
    self.wait_request_is_finished(kivy_clock, req)
    processed_queue = _queue.queue
    assert len(processed_queue) == 1
    self._ensure_called_from_thread(processed_queue)
    self._check_queue_values(processed_queue[0], 'finish')