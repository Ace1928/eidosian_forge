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
def _build_youtube_client(self, creds: Any) -> Any:
    try:
        from googleapiclient.discovery import build
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError:
        raise ImportError('You must run`pip install --upgrade google-api-python-client google-auth-httplib2 google-auth-oauthlib youtube-transcript-api` to use the Google Drive loader')
    return build('youtube', 'v3', credentials=creds)