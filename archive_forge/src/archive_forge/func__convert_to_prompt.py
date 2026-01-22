from __future__ import annotations
import base64
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Union, cast
from urllib.parse import urlparse
import requests
from langchain_core._api.deprecation import deprecated
from langchain_core.callbacks import (
from langchain_core.language_models.chat_models import (
from langchain_core.messages import (
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.pydantic_v1 import root_validator
from langchain_community.llms.vertexai import (
from langchain_community.utilities.vertexai import (
def _convert_to_prompt(part: Union[str, Dict]) -> Part:
    if isinstance(part, str):
        return Part.from_text(part)
    if not isinstance(part, Dict):
        raise ValueError(f"Message's content is expected to be a dict, got {type(part)}!")
    if part['type'] == 'text':
        return Part.from_text(part['text'])
    elif part['type'] == 'image_url':
        path = part['image_url']['url']
        if path.startswith('gs://'):
            image = load_image_from_gcs(path=path, project=project)
        elif path.startswith('data:image/'):
            encoded: Any = re.search('data:image/\\w{2,4};base64,(.*)', path)
            if encoded:
                encoded = encoded.group(1)
            else:
                raise ValueError('Invalid image uri. It should be in the format data:image/<image_type>;base64,<base64_encoded_image>.')
            image = Image.from_bytes(base64.b64decode(encoded))
        elif _is_url(path):
            response = requests.get(path)
            response.raise_for_status()
            image = Image.from_bytes(response.content)
        else:
            image = Image.load_from_file(path)
    else:
        raise ValueError('Only text and image_url types are supported!')
    return Part.from_image(image)