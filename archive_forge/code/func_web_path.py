import asyncio
import logging
import warnings
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union
import aiohttp
import requests
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
@property
def web_path(self) -> str:
    if len(self.web_paths) > 1:
        raise ValueError('Multiple webpaths found.')
    return self.web_paths[0]