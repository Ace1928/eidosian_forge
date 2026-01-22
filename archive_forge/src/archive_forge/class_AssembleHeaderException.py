import base64
import hashlib
import hmac
import json
import logging
from datetime import datetime
from time import mktime
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode
from wsgiref.handlers import format_date_time
import numpy as np
import requests
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, SecretStr, root_validator
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env
from numpy import ndarray
class AssembleHeaderException(Exception):

    def __init__(self, msg: str) -> None:
        self.message = msg