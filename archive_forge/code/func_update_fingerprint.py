import inspect
import os
import random
import shutil
import tempfile
import weakref
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import xxhash
from . import config
from .naming import INVALID_WINDOWS_CHARACTERS_IN_PATH
from .utils._dill import dumps
from .utils.deprecation_utils import deprecated
from .utils.logging import get_logger
def update_fingerprint(fingerprint, transform, transform_args):
    global fingerprint_warnings
    hasher = Hasher()
    hasher.update(fingerprint)
    try:
        hasher.update(transform)
    except:
        if _CACHING_ENABLED:
            if not fingerprint_warnings.get('update_fingerprint_transform_hash_failed', False):
                logger.warning(f"Transform {transform} couldn't be hashed properly, a random hash was used instead. Make sure your transforms and parameters are serializable with pickle or dill for the dataset fingerprinting and caching to work. If you reuse this transform, the caching mechanism will consider it to be different from the previous calls and recompute everything. This warning is only showed once. Subsequent hashing failures won't be showed.")
                fingerprint_warnings['update_fingerprint_transform_hash_failed'] = True
            else:
                logger.info(f"Transform {transform} couldn't be hashed properly, a random hash was used instead.")
        else:
            logger.info(f"Transform {transform} couldn't be hashed properly, a random hash was used instead. This doesn't affect caching since it's disabled.")
        return generate_random_fingerprint()
    for key in sorted(transform_args):
        hasher.update(key)
        try:
            hasher.update(transform_args[key])
        except:
            if _CACHING_ENABLED:
                if not fingerprint_warnings.get('update_fingerprint_transform_hash_failed', False):
                    logger.warning(f"Parameter '{key}'={transform_args[key]} of the transform {transform} couldn't be hashed properly, a random hash was used instead. Make sure your transforms and parameters are serializable with pickle or dill for the dataset fingerprinting and caching to work. If you reuse this transform, the caching mechanism will consider it to be different from the previous calls and recompute everything. This warning is only showed once. Subsequent hashing failures won't be showed.")
                    fingerprint_warnings['update_fingerprint_transform_hash_failed'] = True
                else:
                    logger.info(f"Parameter '{key}'={transform_args[key]} of the transform {transform} couldn't be hashed properly, a random hash was used instead.")
            else:
                logger.info(f"Parameter '{key}'={transform_args[key]} of the transform {transform} couldn't be hashed properly, a random hash was used instead. This doesn't affect caching since it's disabled.")
            return generate_random_fingerprint()
    return hasher.hexdigest()