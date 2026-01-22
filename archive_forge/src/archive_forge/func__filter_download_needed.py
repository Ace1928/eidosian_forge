import hashlib
import json
import os
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Optional
from urllib.request import urlretrieve
from pyproj._sync import get_proj_endpoint
from pyproj.aoi import BBox
from pyproj.datadir import get_data_dir, get_user_data_dir
def _filter_download_needed(feature: dict[str, Any]) -> bool:
    """
    Filter grids so only those that need to be downloaded are included.
    """
    properties = feature.get('properties')
    if not properties:
        return False
    filename = properties.get('name')
    if not filename:
        return False
    return _is_download_needed(filename)