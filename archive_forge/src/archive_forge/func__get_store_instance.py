import base64
import json
import logging
import os
import platform
from typing import Any, Dict, Mapping, Optional, Tuple, Union
import dockerpycreds  # type: ignore
def _get_store_instance(self, name: str) -> 'dockerpycreds.Store':
    if name not in self._stores:
        self._stores[name] = dockerpycreds.Store(name, environment=self._credstore_env)
    return self._stores[name]