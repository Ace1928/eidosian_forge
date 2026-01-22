import textwrap
from typing import Dict, List, Optional
import bq_flags
from utils import bq_logging
class BigqueryError(Exception):
    """Class to represent a BigQuery error."""