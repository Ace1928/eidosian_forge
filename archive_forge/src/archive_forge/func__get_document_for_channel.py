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
def _get_document_for_channel(self, channel: str, **kwargs: Any) -> List[Document]:
    try:
        from youtube_transcript_api import NoTranscriptFound, TranscriptsDisabled
    except ImportError:
        raise ImportError('You must run`pip install --upgrade youtube-transcript-api` to use the youtube loader')
    channel_id = self._get_channel_id(channel)
    request = self.youtube_client.search().list(part='id,snippet', channelId=channel_id, maxResults=50)
    video_ids = []
    while request is not None:
        response = request.execute()
        for item in response['items']:
            if not item['id'].get('videoId'):
                continue
            meta_data = {'videoId': item['id']['videoId']}
            if self.add_video_info:
                item['snippet'].pop('thumbnails')
                meta_data.update(item['snippet'])
            try:
                page_content = self._get_transcripe_for_video_id(item['id']['videoId'])
                video_ids.append(Document(page_content=page_content, metadata=meta_data))
            except (TranscriptsDisabled, NoTranscriptFound) as e:
                if self.continue_on_failure:
                    logger.error('Error fetching transscript ' + f' {item['id']['videoId']}, exception: {e}')
                else:
                    raise e
                pass
        request = self.youtube_client.search().list_next(request, response)
    return video_ids