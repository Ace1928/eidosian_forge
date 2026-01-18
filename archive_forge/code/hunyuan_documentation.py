import base64
import hashlib
import hmac
import json
import logging
import time
from typing import Any, Dict, Iterator, List, Mapping, Optional, Type
from urllib.parse import urlparse
import requests
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import (
from langchain_core.messages import (
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.pydantic_v1 import Field, SecretStr, root_validator
from langchain_core.utils import (
Get the default parameters for calling Hunyuan API.