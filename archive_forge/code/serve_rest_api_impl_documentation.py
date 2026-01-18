import json
import asyncio
import logging
import dataclasses
from functools import wraps
from typing import Union
import aiohttp
from aiohttp.web import Request, Response
import ray
from ray.exceptions import RayTaskError
from ray.dashboard.modules.version import (
import ray.dashboard.utils as dashboard_utils
import ray.dashboard.optional_utils as optional_utils
import ray.dashboard.optional_utils as dashboard_optional_utils
from ray._private.pydantic_compat import ValidationError
Gets the ServeController to the this cluster's Serve app.

            return: If Serve is running on this Ray cluster, returns a client to
                the Serve controller. If Serve is not running, returns None.
            