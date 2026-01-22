from concurrent import futures
import datetime
import json
import logging
import os
import time
import urllib
from absl import flags
def _full_name(self):
    """Returns the full Cloud name for this TPU."""
    return 'projects/%s/locations/%s/nodes/%s' % (self._project, self._zone, self._tpu)