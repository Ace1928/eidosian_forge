import logging
import os
import subprocess
from typing import Iterator, List, Optional
from google.oauth2 import credentials as google_oauth2
import bq_auth_flags
import bq_flags
import bq_utils
from utils import bq_error
def _GetReauthMessage() -> str:
    gcloud_command = '$ gcloud auth login' + (' --enable-gdrive-access' if bq_flags.ENABLE_GDRIVE.value else '')
    return 'To re-authenticate, run:\n\n%s' % gcloud_command