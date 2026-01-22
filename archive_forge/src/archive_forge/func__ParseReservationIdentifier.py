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
def _ParseReservationIdentifier(identifier: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Parses the reservation identifier string into its components.

  Args:
    identifier: String specifying the reservation identifier in the format
      "project_id:reservation_id", "project_id:location.reservation_id", or
      "reservation_id".

  Returns:
    A tuple of three elements: containing project_id, location, and
    reservation_id. If an element is not found, it is represented by None.

  Raises:
    bq_error.BigqueryError: if the identifier could not be parsed.
  """
    pattern = re.compile('\n  ^((?P<project_id>[\\w:\\-.]*[\\w:\\-]+):)?\n  ((?P<location>[\\w\\-]+)\\.)?\n  (?P<reservation_id>[\\w\\-]*)$\n  ', re.X)
    match = re.search(pattern, identifier)
    if not match:
        raise bq_error.BigqueryError('Could not parse reservation identifier: %s' % identifier)
    project_id = match.groupdict().get('project_id', None)
    location = match.groupdict().get('location', None)
    reservation_id = match.groupdict().get('reservation_id', None)
    return (project_id, location, reservation_id)