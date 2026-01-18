from concurrent import futures
import datetime
import json
import logging
import os
import time
import urllib
from absl import flags
def wait_for_healthy(self, timeout_s=1200, interval=30):
    """Wait for TPU to become healthy or raise error if timeout reached.

    Args:
      timeout_s (int): The timeout in seconds for waiting TPU to become healthy.
      interval (int): The interval in seconds to poll the TPU for health.

    Raises:
      RuntimeError: If the TPU doesn't become healthy by the timeout.
    """
    timeout = time.time() + timeout_s
    while self.health() != 'HEALTHY':
        logging.warning('Waiting for TPU "%s" with state "%s" and health "%s" to become healthy', self.name(), self.state(), self.health())
        if time.time() + interval > timeout:
            raise RuntimeError('Timed out waiting for TPU "%s" to become healthy' % self.name())
        time.sleep(interval)
    logging.warning('TPU "%s" is healthy.', self.name())