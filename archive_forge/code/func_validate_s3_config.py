import os
import json
import pathlib
from typing import Optional, Union, Dict, Any
from lazyops.types.models import BaseSettings, validator
from lazyops.types.classprops import lazyproperty
from lazyops.imports._fileio import (
@validator('s3_config', pre=True)
def validate_s3_config(cls, v):
    if v is None:
        return {}
    return json.loads(v) if isinstance(v, str) else v