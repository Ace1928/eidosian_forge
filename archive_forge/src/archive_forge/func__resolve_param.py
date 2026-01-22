import logging
from enum import Enum
from io import BytesIO
from typing import Any, Callable, Dict, Iterator, List, Optional, Union
import requests
from langchain_core.documents import Document
from tenacity import (
from langchain_community.document_loaders.base import BaseLoader
def _resolve_param(self, param_name: str, kwargs: Any) -> Any:
    return kwargs[param_name] if param_name in kwargs else getattr(self, param_name)