import io
import urllib3
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.utils import logger
from sentry_sdk.envelope import Envelope
class SpotlightClient(object):

    def __init__(self, url):
        self.url = url
        self.http = urllib3.PoolManager()
        self.tries = 0

    def capture_envelope(self, envelope):
        if self.tries > 3:
            logger.warning('Too many errors sending to Spotlight, stop sending events there.')
            return
        body = io.BytesIO()
        envelope.serialize_into(body)
        try:
            req = self.http.request(url=self.url, body=body.getvalue(), method='POST', headers={'Content-Type': 'application/x-sentry-envelope'})
            req.close()
        except Exception as e:
            self.tries += 1
            logger.warning(str(e))