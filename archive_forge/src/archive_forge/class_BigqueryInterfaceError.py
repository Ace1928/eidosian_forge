import textwrap
from typing import Dict, List, Optional
import bq_flags
from utils import bq_logging
class BigqueryInterfaceError(BigqueryError):
    """Response from server missing required fields."""