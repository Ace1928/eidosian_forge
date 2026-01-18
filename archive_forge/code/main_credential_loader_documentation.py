import logging
from typing import Union
from absl import app
from google.auth.compute_engine import credentials as compute_engine
from google.oauth2 import credentials as google_oauth2
from google.oauth2 import service_account
import bq_auth_flags
import bq_flags
import bq_utils
from auth import gcloud_credential_loader
from utils import bq_error
Returns credentials based on BQ CLI auth flags.

  Returns: An OAuth2, compute engine, or service account credentials objects
  based on BQ CLI auth flag values.

  Raises:
  app.UsageError, invalid flag values.
  bq_error.BigqueryError, error getting credentials.
  