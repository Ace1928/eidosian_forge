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
def validate_endpoint(log_deprecation_warning: bool):

    def decorator(func):

        @wraps(func)
        async def check(self, *args, **kwargs):
            try:
                from ray import serve
                if log_deprecation_warning:
                    logger.info("The Serve REST API on the dashboard agent is deprecated. Send requests to the Serve REST API directly to the dashboard instead. If you're using default ports, this means you should send the request to the same route on port 8265 instead of 52365.")
            except ImportError:
                return Response(status=501, text='Serve dependencies are not installed. Please run `pip install "ray[serve]"`.')
            return await func(self, *args, **kwargs)
        return check
    return decorator