import requests
import aiohttp
import simdjson as json
from lazyops.utils import timed_cache
from lazyops.serializers import async_cache
from ._base import lazyclass, dataclass, List, Union, Any, Dict
from .tfserving_pb2 import TFSModelConfig, TFSConfig
def validate_models(self):
    for model in list(self.available_models):
        if not self.endpoints[model].is_alive:
            self.available_models.remove(model)
            _ = self.endpoints.pop(model)
    if self.default_model not in self.available_models:
        self.default_model = self.available_models[0]