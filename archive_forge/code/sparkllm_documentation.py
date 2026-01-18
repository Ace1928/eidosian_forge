from __future__ import annotations
import base64
import hashlib
import hmac
import json
import logging
import queue
import threading
from datetime import datetime
from queue import Queue
from time import mktime
from typing import Any, Dict, Generator, Iterator, List, Optional
from urllib.parse import urlencode, urlparse, urlunparse
from wsgiref.handlers import format_date_time
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
from langchain_core.pydantic_v1 import Field, root_validator
from langchain_core.utils import get_from_dict_or_env

        Generate a request url with an api key and an api secret.
        