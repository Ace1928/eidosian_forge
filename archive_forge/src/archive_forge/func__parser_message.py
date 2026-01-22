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
@staticmethod
def _parser_message(message: str) -> Optional[ndarray]:
    data = json.loads(message)
    code = data['header']['code']
    if code != 0:
        logger.warning(f'Request error: {code}, {data}')
        return None
    else:
        text_base = data['payload']['feature']['text']
        text_data = base64.b64decode(text_base)
        dt = np.dtype(np.float32)
        dt = dt.newbyteorder('<')
        text = np.frombuffer(text_data, dtype=dt)
        if len(text) > 2560:
            array = text[:2560]
        else:
            array = text
        return array