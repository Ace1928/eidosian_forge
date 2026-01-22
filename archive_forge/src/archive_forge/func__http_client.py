import re
from contextlib import contextmanager
from dataclasses import dataclass
from json.decoder import JSONDecodeError
from typing import Iterator, Any, Dict, Union, Tuple, Optional, List, cast
import httpx
from qcs_api_client.client import QCSClientConfiguration
from pyquil.api._errors import ApiError, UnknownApiError, TooManyQubitsError, error_mapping
@contextmanager
def _http_client(self) -> Iterator[httpx.Client]:
    with httpx.Client(base_url=self.base_url, timeout=self.timeout) as client:
        yield client