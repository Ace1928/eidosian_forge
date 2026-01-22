from concurrent import futures
import datetime
import json
import logging
import os
import time
import urllib
from absl import flags
def _tpu_service(self):
    """Creates a new Cloud TPU API object.

    This works around an issue where the underlying HTTP connection sometimes
    times out when the script has been running for too long. Other methods in
    this object call this method to get a new API object whenever they need
    to communicate with the Cloud API.

    Raises:
      RuntimeError: If the dependent Python packages are missing.

    Returns:
      A Google Cloud TPU API object.
    """
    if self._service:
        return self._service
    if not _GOOGLE_API_CLIENT_INSTALLED:
        raise RuntimeError('Missing runtime dependency on the Google API client. Run `pip install cloud-tpu-client` to fix.')
    credentials = self._credentials
    if credentials is None or credentials == 'default':
        credentials = client.GoogleCredentials.get_application_default()
    if self._discovery_url:
        return discovery.build('tpu', 'v1', credentials=credentials, discoveryServiceUrl=self._discovery_url, cache_discovery=False)
    else:
        return discovery.build('tpu', 'v1', credentials=credentials, cache_discovery=False)