from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc
import collections
import datetime
import hashlib
import json
import logging
import os
import random
import re
import string
import sys
import time
from typing import Dict, List, NamedTuple, Optional, Tuple, Union
from absl import flags
import googleapiclient
import httplib2
from utils import bq_api_utils
from utils import bq_error
from utils import bq_id_utils
from utils import bq_processor_utils
def ParseReservationPath(path: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Parses the reservation path string into its components.

  Args:
    path: String specifying the reservation path in the format
      projects/<project_id>/locations/<location>/reservations/<reservation_id>
      or
      projects/<project_id>/locations/<location>/biReservation

  Returns:
    A tuple of three elements: containing project_id, location and
    reservation_id. If an element is not found, it is represented by None.

  Raises:
    bq_error.BigqueryError: if the path could not be parsed.
  """
    pattern = re.compile('^projects/(?P<project_id>[\\w:\\-.]*[\\w:\\-]+)?' + '/locations/(?P<location>[\\w\\-]+)?' + '/(reservations/(?P<reservation_id>[\\w\\-/]+)' + '|(?P<bi_id>biReservation)' + ')$', re.X)
    match = re.search(pattern, path)
    if not match:
        raise bq_error.BigqueryError('Could not parse reservation path: %s' % path)
    group = lambda key: match.groupdict().get(key, None)
    project_id = group('project_id')
    location = group('location')
    reservation_id = group('reservation_id') or group('bi_id')
    return (project_id, location, reservation_id)