import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Dict, Optional
from .inference._client import InferenceClient
from .inference._generated._async_client import AsyncInferenceClient
from .utils import logging, parse_datetime
Populate fields from raw dictionary.

        Called in __post_init__ + each time the Inference Endpoint is updated.
        