import abc
import fnmatch
import glob
import logging
import os
import queue
import time
from typing import TYPE_CHECKING, Any, Mapping, MutableMapping, MutableSet, Optional
from wandb import util
from wandb.sdk.interface.interface import GlobStr
from wandb.sdk.lib.paths import LogicalPath
Get or create an event handler for a particular file.

        file_path: the file's actual path
        save_name: its path relative to the run directory (aka the watch directory)
        