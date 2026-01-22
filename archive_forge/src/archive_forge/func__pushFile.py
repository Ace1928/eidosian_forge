import asyncio
import base64
import logging
import mimetypes
import os
from typing import Any, Dict, Optional, Type, Union
import requests
from langchain_core.callbacks import (
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool
def _pushFile(self, id: str, content_path: str) -> str:
    with open(content_path, 'rb') as source_file:
        response = requests.post(self._config['BACKEND'] + '/processing/upload', headers={'content-type': mimetypes.guess_type(content_path)[0] or 'application/octet-stream', 'x-stf-nuakey': 'Bearer ' + self._config['NUA_KEY']}, data=source_file.read())
        if response.status_code != 200:
            logger.info(f'Error uploading {content_path}: {response.status_code} {response.text}')
            return ''
        else:
            field = {'filefield': {'file': f'{response.text}'}, 'processing_options': {'ml_text': self._config['enable_ml']}}
            return self._pushField(id, field)