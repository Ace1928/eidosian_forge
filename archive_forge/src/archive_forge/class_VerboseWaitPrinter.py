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
class VerboseWaitPrinter(WaitPrinterHelper):
    """A WaitPrinter that prints every update."""

    def __init__(self):
        self.output_token = None

    def Print(self, job_id, wait_time, status):
        self.print_on_done = True
        self.output_token = _OverwriteCurrentLine('Waiting on %s ... (%ds) Current status: %-7s' % (job_id, wait_time, status), self.output_token)