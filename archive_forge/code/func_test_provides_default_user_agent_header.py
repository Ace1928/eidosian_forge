import pytest
from dummyserver.testcase import (
from urllib3 import HTTPConnectionPool
from urllib3.util import SKIP_HEADER
from urllib3.util.retry import Retry
def test_provides_default_user_agent_header(self):
    self.start_chunked_handler()
    chunks = ['foo', 'bar', '', 'bazzzzzzzzzzzzzzzzzzzzzz']
    with HTTPConnectionPool(self.host, self.port, retries=False) as pool:
        pool.urlopen('GET', '/', body=chunks, chunked=True)
        ua_headers = self._get_header_lines(b'user-agent')
        assert len(ua_headers) == 1