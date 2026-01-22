import abc
import logging
import os
import random
import shutil
import time
import urllib
import uuid
from collections import namedtuple
from typing import IO, List, Optional, Tuple
import ray
from ray._private.ray_constants import DEFAULT_OBJECT_PREFIX
from ray._raylet import ObjectRef
def _destroy_external_storage(self, directory_path):
    while os.path.isdir(directory_path):
        try:
            shutil.rmtree(directory_path)
        except FileNotFoundError:
            pass
        except Exception:
            logger.exception('Error cleaning up spill files. You might still have remaining spilled objects inside `ray_spilled_objects` directory.')
            break