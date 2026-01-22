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
class WaitPrinter:
    """Base class that defines the WaitPrinter interface."""

    def Print(self, job_id, wait_time, status):
        """Prints status for the current job we are waiting on.

    Args:
      job_id: the identifier for this job.
      wait_time: the number of seconds we have been waiting so far.
      status: the status of the job we are waiting for.
    """
        raise NotImplementedError('Subclass must implement Print')

    def Done(self):
        """Waiting is done and no more Print calls will be made.

    This function should handle the case of Print not being called.
    """
        raise NotImplementedError('Subclass must implement Done')