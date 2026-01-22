from __future__ import annotations
import logging
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
from urllib.parse import parse_qs, urlparse
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import root_validator
from langchain_core.pydantic_v1.dataclasses import dataclass
from langchain_community.document_loaders.base import BaseLoader
def _parse_video_id(url: str) -> Optional[str]:
    """Parse a youtube url and return the video id if valid, otherwise None."""
    parsed_url = urlparse(url)
    if parsed_url.scheme not in ALLOWED_SCHEMAS:
        return None
    if parsed_url.netloc not in ALLOWED_NETLOCK:
        return None
    path = parsed_url.path
    if path.endswith('/watch'):
        query = parsed_url.query
        parsed_query = parse_qs(query)
        if 'v' in parsed_query:
            ids = parsed_query['v']
            video_id = ids if isinstance(ids, str) else ids[0]
        else:
            return None
    else:
        path = parsed_url.path.lstrip('/')
        video_id = path.split('/')[-1]
    if len(video_id) != 11:
        return None
    return video_id