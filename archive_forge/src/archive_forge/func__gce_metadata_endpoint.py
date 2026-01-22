from concurrent import futures
import datetime
import json
import logging
import os
import time
import urllib
from absl import flags
def _gce_metadata_endpoint():
    endpoint = os.environ.get(_GCE_METADATA_ENDPOINT_ENV_VARIABLE)
    if not endpoint:
        endpoint = os.environ.get(_GCE_METADATA_URL_ENV_VARIABLE, 'metadata.google.internal')
    return 'http://' + endpoint