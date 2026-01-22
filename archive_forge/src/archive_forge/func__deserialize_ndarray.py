import base64
from collections import OrderedDict
import importlib
import io
import zlib
from typing import Any, Dict, Optional, Sequence, Type, Union
import numpy as np
import ray
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.utils.gym import try_import_gymnasium_and_gym
from ray.rllib.utils.error import NotSerializable
from ray.rllib.utils.spaces.flexdict import FlexDict
from ray.rllib.utils.spaces.repeated import Repeated
from ray.rllib.utils.spaces.simplex import Simplex
def _deserialize_ndarray(b64_string: str) -> np.ndarray:
    """Unpack b64 escaped string into numpy ndarray.

    This function assumes the unescaped bytes are of npy format.

    Args:
        b64_string: Base64 escaped string.

    Returns:
        numpy ndarray.
    """
    return np.load(io.BytesIO(zlib.decompress(base64.b64decode(b64_string))))