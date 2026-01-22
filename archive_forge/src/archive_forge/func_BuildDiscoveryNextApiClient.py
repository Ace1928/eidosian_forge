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
def BuildDiscoveryNextApiClient(self) -> discovery.Resource:
    """Builds and returns BigQuery API client from discovery_next document."""
    http = self.GetAuthorizedHttp(self.credentials, self.GetHttp())
    bigquery_model = bigquery_http.BigqueryModel(trace=self.trace, quota_project_id=bq_utils.GetEffectiveQuotaProjectIDForHTTPHeader(self.quota_project_id, self.use_google_auth, self.credentials))
    bq_request_builder = bigquery_http.BigqueryHttp.Factory(bigquery_model, self.use_google_auth)
    models_doc = None
    try:
        models_doc = discovery_document_loader.load_local_discovery_doc(discovery_document_loader.DISCOVERY_NEXT_BIGQUERY)
        models_doc = self.OverrideEndpoint(discovery_document=models_doc, service=Service.BIGQUERY)
    except (bq_error.BigqueryClientError, FileNotFoundError) as e:
        logging.warning('Failed to load discovery doc from local files: %s', e)
        raise
    try:
        return discovery.build_from_document(models_doc, http=http, model=bigquery_model, requestBuilder=bq_request_builder)
    except Exception:
        logging.error('Error building from models document: %s', models_doc)
        raise