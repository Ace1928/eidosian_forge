from __future__ import annotations
import logging
from typing import Dict, List, Literal, Optional
import requests
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import Field, root_validator, validator
from langchain_community.tools.edenai.edenai_base_tool import EdenaiTool
def _download_wav(self, url: str, save_path: str) -> None:
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(response.content)
    else:
        raise ValueError('Error while downloading wav file')