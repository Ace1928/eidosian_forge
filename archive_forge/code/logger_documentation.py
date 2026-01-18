import abc
import json
import logging
import os
import pyarrow
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Set, Type
import yaml
from ray.air._internal.json import SafeFallbackEncoder
from ray.tune.callback import Callback
from ray.util.annotations import Deprecated, DeveloperAPI, PublicAPI
Handle logging when a trial ends.

        Args:
            trial: Trial object.
            failed: True if the Trial finished gracefully, False if
                it failed (e.g. when it raised an exception).
        