import collections
import dataclasses
import json
import logging
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
from wandb.sdk.lib import telemetry
from .aggregators import aggregate_mean
from .asset_registry import asset_registry
from .interfaces import Interface, Metric, MetricsMonitor
def _is_matching_entry(self, entry: dict) -> bool:
    """Check if the entry should be saved.

        Checks if the pid in the entry matches the pid of the process.
        If not (as in the case of multi-process training with torchrun),
        checks if the LOCAL_RANK environment variable is set.

        todo: add matching by neuron_runtime_tag
        """
    return int(entry['pid']) == int(self.pid) or 'LOCAL_RANK' in os.environ