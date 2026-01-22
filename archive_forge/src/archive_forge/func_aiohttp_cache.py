import asyncio
import collections
import functools
import inspect
import json
import logging
import os
import time
import traceback
from collections import namedtuple
from typing import Any, Callable
from aiohttp.web import Response
import ray
import ray.dashboard.consts as dashboard_consts
from ray._private.ray_constants import RAY_INTERNAL_DASHBOARD_NAMESPACE, env_bool
from ray.dashboard.optional_deps import PathLike, RouteDef, aiohttp, hdrs
from ray.dashboard.utils import CustomEncoder, to_google_style
def aiohttp_cache(ttl_seconds=dashboard_consts.AIOHTTP_CACHE_TTL_SECONDS, maxsize=dashboard_consts.AIOHTTP_CACHE_MAX_SIZE, enable=not env_bool(dashboard_consts.AIOHTTP_CACHE_DISABLE_ENVIRONMENT_KEY, False)):
    assert maxsize > 0
    cache = collections.OrderedDict()

    def _wrapper(handler):
        if enable:

            @functools.wraps(handler)
            async def _cache_handler(*args) -> aiohttp.web.Response:
                req = args[-1]
                if req.method in _AIOHTTP_CACHE_NOBODY_METHODS:
                    key = req.path_qs
                else:
                    key = (req.path_qs, await req.read())
                value = cache.get(key)
                if value is not None:
                    cache.move_to_end(key)
                    if not value.task.done() or value.expiration >= time.time():
                        return aiohttp.web.Response(**value.data)

                def _update_cache(task):
                    try:
                        response = task.result()
                    except Exception:
                        response = rest_response(success=False, message=traceback.format_exc())
                    data = {'status': response.status, 'headers': dict(response.headers), 'body': response.body}
                    cache[key] = _AiohttpCacheValue(data, time.time() + ttl_seconds, task)
                    cache.move_to_end(key)
                    if len(cache) > maxsize:
                        cache.popitem(last=False)
                    return response
                task = create_task(handler(*args))
                task.add_done_callback(_update_cache)
                if value is None:
                    return await task
                else:
                    return aiohttp.web.Response(**value.data)
            suffix = f'[cache ttl={ttl_seconds}, max_size={maxsize}]'
            _cache_handler.__name__ += suffix
            _cache_handler.__qualname__ += suffix
            return _cache_handler
        else:
            return handler
    if inspect.iscoroutinefunction(ttl_seconds):
        target_func = ttl_seconds
        ttl_seconds = dashboard_consts.AIOHTTP_CACHE_TTL_SECONDS
        return _wrapper(target_func)
    else:
        return _wrapper