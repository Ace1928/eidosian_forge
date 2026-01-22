import os
import re
import time
from enum import Enum
from typing import List, Optional
import requests
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
@staticmethod
def _detect_value_type(tokenId: str) -> str:
    if isinstance(tokenId, int):
        return 'int'
    elif tokenId.startswith('0x'):
        return 'hex_0x'
    elif tokenId.startswith('0xbf'):
        return 'hex_0xbf'
    else:
        return 'hex_0xbf'