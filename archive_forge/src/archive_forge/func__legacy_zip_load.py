import contextlib
import errno
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import torch
import uuid
import warnings
import zipfile
from pathlib import Path
from typing import Dict, Optional, Any
from urllib.error import HTTPError, URLError
from urllib.request import urlopen, Request
from urllib.parse import urlparse  # noqa: F401
from torch.serialization import MAP_LOCATION
def _legacy_zip_load(filename: str, model_dir: str, map_location: MAP_LOCATION, weights_only: bool) -> Dict[str, Any]:
    warnings.warn('Falling back to the old format < 1.6. This support will be deprecated in favor of default zipfile format introduced in 1.6. Please redo torch.save() to save it in the new zipfile format.')
    with zipfile.ZipFile(filename) as f:
        members = f.infolist()
        if len(members) != 1:
            raise RuntimeError('Only one file(not dir) is allowed in the zipfile')
        f.extractall(model_dir)
        extraced_name = members[0].filename
        extracted_file = os.path.join(model_dir, extraced_name)
    return torch.load(extracted_file, map_location=map_location, weights_only=weights_only)