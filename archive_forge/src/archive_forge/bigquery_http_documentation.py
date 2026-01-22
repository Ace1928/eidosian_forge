from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import logging
from typing import Optional
from absl import flags
import googleapiclient
from googleapiclient import http as http_request
from googleapiclient import model
import httplib2
import bq_utils
from clients import utils as bq_client_utils
Raises a BigQueryError given a non-HttpError.