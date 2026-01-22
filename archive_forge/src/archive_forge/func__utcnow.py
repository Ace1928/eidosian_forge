from concurrent import futures
import datetime
import json
import logging
import os
import time
import urllib
from absl import flags
def _utcnow():
    """A wrapper function around datetime.datetime.utcnow.

  This function is created for unit testing purpose. It's not easy to do
  StubOutWithMock with datetime.datetime package.

  Returns:
    datetime.datetime
  """
    return datetime.datetime.utcnow()