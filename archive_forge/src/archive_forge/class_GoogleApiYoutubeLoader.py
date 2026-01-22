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
@dataclass
class GoogleApiYoutubeLoader(BaseLoader):
    """Load all Videos from a `YouTube` Channel.

    To use, you should have the ``googleapiclient,youtube_transcript_api``
    python package installed.
    As the service needs a google_api_client, you first have to initialize
    the GoogleApiClient.

    Additionally you have to either provide a channel name or a list of videoids
    "https://developers.google.com/docs/api/quickstart/python"



    Example:
        .. code-block:: python

            from langchain_community.document_loaders import GoogleApiClient
            from langchain_community.document_loaders import GoogleApiYoutubeLoader
            google_api_client = GoogleApiClient(
                service_account_path=Path("path_to_your_sec_file.json")
            )
            loader = GoogleApiYoutubeLoader(
                google_api_client=google_api_client,
                channel_name = "CodeAesthetic"
            )
            load.load()

    """
    google_api_client: GoogleApiClient
    channel_name: Optional[str] = None
    video_ids: Optional[List[str]] = None
    add_video_info: bool = True
    captions_language: str = 'en'
    continue_on_failure: bool = False

    def __post_init__(self) -> None:
        self.youtube_client = self._build_youtube_client(self.google_api_client.creds)

    def _build_youtube_client(self, creds: Any) -> Any:
        try:
            from googleapiclient.discovery import build
            from youtube_transcript_api import YouTubeTranscriptApi
        except ImportError:
            raise ImportError('You must run`pip install --upgrade google-api-python-client google-auth-httplib2 google-auth-oauthlib youtube-transcript-api` to use the Google Drive loader')
        return build('youtube', 'v3', credentials=creds)

    @root_validator
    def validate_channel_or_videoIds_is_set(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that either folder_id or document_ids is set, but not both."""
        if not values.get('channel_name') and (not values.get('video_ids')):
            raise ValueError('Must specify either channel_name or video_ids')
        return values

    def _get_transcripe_for_video_id(self, video_id: str) -> str:
        from youtube_transcript_api import NoTranscriptFound, YouTubeTranscriptApi
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        try:
            transcript = transcript_list.find_transcript([self.captions_language])
        except NoTranscriptFound:
            for available_transcript in transcript_list:
                transcript = available_transcript.translate(self.captions_language)
                continue
        transcript_pieces = transcript.fetch()
        return ' '.join([t['text'].strip(' ') for t in transcript_pieces])

    def _get_document_for_video_id(self, video_id: str, **kwargs: Any) -> Document:
        captions = self._get_transcripe_for_video_id(video_id)
        video_response = self.youtube_client.videos().list(part='id,snippet', id=video_id).execute()
        return Document(page_content=captions, metadata=video_response.get('items')[0])

    def _get_channel_id(self, channel_name: str) -> str:
        request = self.youtube_client.search().list(part='id', q=channel_name, type='channel', maxResults=1)
        response = request.execute()
        channel_id = response['items'][0]['id']['channelId']
        return channel_id

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

    def load(self) -> List[Document]:
        """Load documents."""
        document_list = []
        if self.channel_name:
            document_list.extend(self._get_document_for_channel(self.channel_name))
        elif self.video_ids:
            document_list.extend([self._get_document_for_video_id(video_id) for video_id in self.video_ids])
        else:
            raise ValueError('Must specify either channel_name or video_ids')
        return document_list