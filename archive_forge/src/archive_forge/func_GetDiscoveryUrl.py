from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import enum
from http import client as http_client_lib
import json
import logging
import re
import sys
import tempfile
import time
import traceback
from typing import Any, Callable, Optional, Union
import urllib
from absl import flags
import googleapiclient
from googleapiclient import discovery
import httplib2
import bq_flags
import bq_utils
from clients import bigquery_http
from clients import utils as bq_client_utils
from discovery_documents import discovery_document_cache
from discovery_documents import discovery_document_loader
from utils import bq_api_utils
from utils import bq_error
def GetDiscoveryUrl(self, service: Service, api_version: str) -> str:
    """Returns the url to the discovery document for bigquery."""
    discovery_url = ''
    if not discovery_url:
        discovery_url = bq_api_utils.get_discovery_url_from_root_url(bq_api_utils.get_tpc_root_url_from_flags(service=service, inputted_flags=bq_flags, local_params=self), api_version=api_version)
    return discovery_url