from typing import Any, Dict, List, Optional, Union
from huggingface_hub import HfFileSystem
from . import config
from .table import CastError
from .utils.track import TrackedIterable, tracked_list, tracked_str
class DatasetsError(Exception):
    """Base class for exceptions in this library."""