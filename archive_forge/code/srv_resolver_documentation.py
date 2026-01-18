from __future__ import annotations
import ipaddress
import random
from typing import Any, Optional, Union
from pymongo.common import CONNECT_TIMEOUT
from pymongo.errors import ConfigurationError
Support for resolving hosts and options from mongodb+srv:// URIs.