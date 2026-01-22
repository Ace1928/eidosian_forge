from typing import Any, Dict, List, Optional, Union
from .. import config
from ..exceptions import DatasetsError
from .file_utils import (
from .logging import get_logger
class DatasetViewerError(DatasetsError):
    """Dataset viewer error.

    Raised when trying to use the dataset viewer HTTP API and when trying to access:
    - a missing dataset, or
    - a private/gated dataset and the user is not authenticated.
    - unavailable /parquet or /info responses
    """