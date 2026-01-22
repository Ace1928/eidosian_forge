import codecs
import copy
import http.client
import json
import logging
import os
import pkgutil
import platform
import sys
import textwrap
import time
import traceback
from typing import Any, Dict, List, Optional, TextIO
from absl import app
from absl import flags
from google.auth import version as google_auth_version
from google.oauth2 import credentials as google_oauth2
import googleapiclient
import httplib2
import oauth2client_4_0.client
import requests
import urllib3
from utils import bq_error
from utils import bq_logging
from pyglib import stringutil
def GetResolvedQuotaProjectID(quota_project_id: Optional[str], fallback_project_id: Optional[str]) -> Optional[str]:
    """Return the final resolved quota project ID after cross-referencing gcloud properties defined in http://google3/third_party/py/googlecloudsdk/core/properties.py;l=1647;rcl=598870349."""
    if not quota_project_id and fallback_project_id:
        return fallback_project_id
    if quota_project_id and quota_project_id in ('CURRENT_PROJECT', 'CURRENT_PROJECT_WITH_FALLBACK'):
        return fallback_project_id
    if 'LEGACY' == quota_project_id:
        return None
    return quota_project_id