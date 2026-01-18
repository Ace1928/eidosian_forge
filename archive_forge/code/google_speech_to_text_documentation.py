from __future__ import annotations
from typing import TYPE_CHECKING, List, Optional
from langchain_core._api.deprecation import deprecated
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
from langchain_community.utilities.vertexai import get_client_info
Transcribes the audio file and loads the transcript into documents.

        It uses the Google Cloud Speech-to-Text API to transcribe the audio file
        and blocks until the transcription is finished.
        