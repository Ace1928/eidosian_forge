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
class JobIdGeneratorRandom(JobIdGenerator):
    """Generates random job id_fallbacks."""

    def Generate(self, job_configuration):
        return 'bqjob_r%08x_%016x' % (random.SystemRandom().randint(0, sys.maxsize), int(time.time() * 1000))