import textwrap
from typing import Dict, List, Optional
import bq_flags
from utils import bq_logging
class BigqueryInvalidQueryError(BigqueryServiceError):
    """The SQL statement is invalid."""