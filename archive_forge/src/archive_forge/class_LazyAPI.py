import requests
import aiohttp
import asyncio
from dataclasses import dataclass
from lazyops.lazyclasses import lazyclass
from typing import List, Dict, Any, Optional
class LazyAPI:

    def __init__(self, config: LazyAPIConfig, *args, **kwargs):
        self.config = config
        self.header = LazyHeader(user=self.config.user, key=self.config.key, token=self.config.token)
        self.api = LazyAPIModel(self.config.url, self.header, self.config.route_config)
        for route_name in list(self.config.route_config.keys()):
            func = lambda route_name=route_name, *args, **kwargs: self.api.get(*args, route=route_name, **kwargs)
            setattr(self, route_name, func)

    def api_call(self, is_async=False, *args, **kwargs):
        return self.api(*args, route=self.config.default_async if is_async else self.config.default_fetch, **kwargs)

    async def async_call(self, *args, **kwargs):
        return await self.api.async_call(*args, route=self.config.default_async, **kwargs)

    def __call__(self, route, *args, **kwargs):
        return self.api(route, *args, **kwargs)

    @classmethod
    def build(cls, config, *args, **kwargs):
        if isinstance(config, LazyAPIConfig):
            pass
        elif isinstance(config, dict):
            config = LazyAPIConfig.from_dict(config)
        elif isinstance(config, str):
            config = LazyAPIConfig.from_json(config)
        return LazyAPI(config, *args, **kwargs)