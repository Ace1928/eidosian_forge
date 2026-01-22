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
def _ParseConnectionIdentifier(identifier: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Parses the connection identifier string into its components.

  Args:
    identifier: String specifying the connection identifier in the format
      "connection_id", "location.connection_id",
      "project_id.location.connection_id"

  Returns:
    A tuple of four elements: containing project_id, location, connection_id
    If an element is not found, it is represented by None.

  Raises:
    bq_error.BigqueryError: if the identifier could not be parsed.
  """
    if not identifier:
        raise bq_error.BigqueryError('Empty connection identifier')
    tokens = identifier.split('.')
    num_tokens = len(tokens)
    if num_tokens > 4:
        raise bq_error.BigqueryError('Could not parse connection identifier: %s' % identifier)
    connection_id = tokens[num_tokens - 1]
    location = tokens[num_tokens - 2] if num_tokens > 1 else None
    project_id = '.'.join(tokens[:num_tokens - 2]) if num_tokens > 2 else None
    return (project_id, location, connection_id)