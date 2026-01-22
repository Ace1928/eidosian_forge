import copy
import glob
import inspect
import logging
import os
import threading
import time
from collections import defaultdict
from datetime import datetime
from numbers import Number
from threading import Thread
from typing import Any, Callable, Dict, List, Optional, Sequence, Type, Union
import numpy as np
import psutil
import ray
from ray.util.annotations import DeveloperAPI, PublicAPI
from ray.air._internal.json import SafeFallbackEncoder  # noqa
from ray.air._internal.util import (  # noqa: F401
from ray._private.dict import (  # noqa: F401
def _detect_checkpoint_function(train_func, abort=False, partial=False):
    """Use checkpointing if any arg has "checkpoint_dir" and args = 2"""
    func_sig = inspect.signature(train_func)
    validated = True
    try:
        if partial:
            func_sig.bind_partial({}, checkpoint_dir='tmp/path')
        else:
            func_sig.bind({}, checkpoint_dir='tmp/path')
    except Exception as e:
        logger.debug(str(e))
        validated = False
    if abort and (not validated):
        func_args = inspect.getfullargspec(train_func).args
        raise ValueError('Provided training function must have 1 `config` argument `func(config)`. Got {}'.format(func_args))
    return validated