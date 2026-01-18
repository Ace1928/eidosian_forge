import base64
import json
from ray import cloudpickle
from enum import Enum, unique
import hashlib
from typing import Dict, Optional, Any, Tuple
from dataclasses import dataclass
import ray
from ray import ObjectRef
from ray._private.utils import get_or_create_event_loop
from ray.util.annotations import PublicAPI
def validate_user_metadata(metadata):
    if metadata is not None:
        if not isinstance(metadata, dict):
            raise ValueError('metadata must be a dict.')
        try:
            json.dumps(metadata)
        except TypeError as e:
            raise ValueError("metadata must be JSON serializable, instead, we got 'TypeError: {}'".format(e))