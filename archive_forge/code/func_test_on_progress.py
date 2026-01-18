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
def test_on_progress(self, kivy_clock):
    _queue = UrlRequestQueue([])
    responses.get(self.url, body='x' * 100, status=200, content_type='text/plain', auto_calculate_content_length=True)
    req = UrlRequest(self.url, on_progress=_queue._on_progress, chunk_size=70, debug=True)
    self.wait_request_is_finished(kivy_clock, req)
    processed_queue = _queue.queue
    assert len(processed_queue) == 4
    self._ensure_called_from_thread(processed_queue)
    self._check_queue_values(processed_queue[0], 'progress')
    self._check_queue_values(processed_queue[1], 'progress')
    self._check_queue_values(processed_queue[2], 'progress')
    self._check_queue_values(processed_queue[3], 'progress')