import logging
import sys
from typing import Optional, TextIO
from absl import flags
from absl import logging as absl_logging
from googleapiclient import model
Safely encode an object as the encoding for sys.stdout.