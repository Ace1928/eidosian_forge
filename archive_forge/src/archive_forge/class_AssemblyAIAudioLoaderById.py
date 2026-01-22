from __future__ import annotations
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Optional, Union
import requests
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
class AssemblyAIAudioLoaderById(BaseLoader):
    """
    Load AssemblyAI audio transcripts.

    It uses the AssemblyAI API to get an existing transcription
    and loads the transcribed text into one or more Documents,
    depending on the specified format.

    """

    def __init__(self, transcript_id: str, api_key: str, transcript_format: TranscriptFormat):
        """
        Initializes the AssemblyAI AssemblyAIAudioLoaderById.

        Args:
            transcript_id: Id of an existing transcription.
            transcript_format: Transcript format to use.
                See class ``TranscriptFormat`` for more info.
            api_key: AssemblyAI API key.
        """
        self.api_key = api_key
        self.transcript_id = transcript_id
        self.transcript_format = transcript_format

    def lazy_load(self) -> Iterator[Document]:
        """Load data into Document objects."""
        HEADERS = {'authorization': self.api_key}
        if self.transcript_format == TranscriptFormat.TEXT:
            try:
                transcript_response = requests.get(f'https://api.assemblyai.com/v2/transcript/{self.transcript_id}', headers=HEADERS)
                transcript_response.raise_for_status()
            except Exception as e:
                print(f'An error occurred: {e}')
                raise
            transcript = transcript_response.json()['text']
            yield Document(page_content=transcript, metadata=transcript_response.json())
        elif self.transcript_format == TranscriptFormat.PARAGRAPHS:
            try:
                paragraphs_response = requests.get(f'https://api.assemblyai.com/v2/transcript/{self.transcript_id}/paragraphs', headers=HEADERS)
                paragraphs_response.raise_for_status()
            except Exception as e:
                print(f'An error occurred: {e}')
                raise
            paragraphs = paragraphs_response.json()['paragraphs']
            for p in paragraphs:
                yield Document(page_content=p['text'], metadata=p)
        elif self.transcript_format == TranscriptFormat.SENTENCES:
            try:
                sentences_response = requests.get(f'https://api.assemblyai.com/v2/transcript/{self.transcript_id}/sentences', headers=HEADERS)
                sentences_response.raise_for_status()
            except Exception as e:
                print(f'An error occurred: {e}')
                raise
            sentences = sentences_response.json()['sentences']
            for s in sentences:
                yield Document(page_content=s['text'], metadata=s)
        elif self.transcript_format == TranscriptFormat.SUBTITLES_SRT:
            try:
                srt_response = requests.get(f'https://api.assemblyai.com/v2/transcript/{self.transcript_id}/srt', headers=HEADERS)
                srt_response.raise_for_status()
            except Exception as e:
                print(f'An error occurred: {e}')
                raise
            srt = srt_response.text
            yield Document(page_content=srt)
        elif self.transcript_format == TranscriptFormat.SUBTITLES_VTT:
            try:
                vtt_response = requests.get(f'https://api.assemblyai.com/v2/transcript/{self.transcript_id}/vtt', headers=HEADERS)
                vtt_response.raise_for_status()
            except Exception as e:
                print(f'An error occurred: {e}')
                raise
            vtt = vtt_response.text
            yield Document(page_content=vtt)
        else:
            raise ValueError('Unknown transcript format.')