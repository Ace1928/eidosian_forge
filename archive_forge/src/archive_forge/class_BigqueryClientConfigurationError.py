import textwrap
from typing import Dict, List, Optional
import bq_flags
from utils import bq_logging
class BigqueryClientConfigurationError(BigqueryClientError):
    """Invalid configuration of BigqueryClient."""