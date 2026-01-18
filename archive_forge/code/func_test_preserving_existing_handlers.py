import io
import logging
import time
import testtools
from testtools import TestCase
from fixtures import (
def test_preserving_existing_handlers(self):
    stream = io.StringIO()
    self.logger.addHandler(logging.StreamHandler(stream))
    self.logger.setLevel(logging.INFO)
    fixture = LogHandler(self.CustomHandler(), nuke_handlers=False)
    with fixture:
        logging.info('message')
    self.assertEqual(['message'], fixture.handler.msgs)
    self.assertEqual('message\n', stream.getvalue())