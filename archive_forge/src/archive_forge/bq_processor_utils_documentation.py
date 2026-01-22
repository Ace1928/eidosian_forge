from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import os
import re
from typing import Any, List, NamedTuple, Optional
from utils import bq_error
from utils import bq_id_utils
Processes the Refresh Window Days flag.

  Args:
    refresh_window_days: The user specified refresh window days.
    data_source_info: The data source of the transfer config.
    items: The body that contains information of all the flags set.
    data_source: The data source of the transfer config.

  Returns:
    items: The body after it has been updated with the
    refresh window days flag.
  Raises:
    bq_error.BigqueryError: If the data source does not support (custom)
      window days.
  