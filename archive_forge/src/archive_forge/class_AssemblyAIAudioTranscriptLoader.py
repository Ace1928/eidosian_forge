from __future__ import annotations
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Optional, Union
import requests
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
class AssemblyAIAudioTranscriptLoader(BaseLoader):
    """Load AssemblyAI audio transcripts.

    It uses the AssemblyAI API to transcribe audio files
    and loads the transcribed text into one or more Documents,
    depending on the specified format.

    To use, you should have the ``assemblyai`` python package installed, and the
    environment variable ``ASSEMBLYAI_API_KEY`` set with your API key.
    Alternatively, the API key can also be passed as an argument.

    Audio files can be specified via an URL or a local file path.
    """

    def __init__(self, file_path: Union[str, Path], *, transcript_format: TranscriptFormat=TranscriptFormat.TEXT, config: Optional[assemblyai.TranscriptionConfig]=None, api_key: Optional[str]=None):
        """
        Initializes the AssemblyAI AudioTranscriptLoader.

        Args:
            file_path: An URL or a local file path.
            transcript_format: Transcript format to use.
                See class ``TranscriptFormat`` for more info.
            config: Transcription options and features. If ``None`` is given,
                the Transcriber's default configuration will be used.
            api_key: AssemblyAI API key.
        """
        try:
            import assemblyai
        except ImportError:
            raise ImportError('Could not import assemblyai python package. Please install it with `pip install assemblyai`.')
        if api_key is not None:
            assemblyai.settings.api_key = api_key
        self.file_path = str(file_path)
        self.transcript_format = transcript_format
        self.transcriber = assemblyai.Transcriber(config=config)

    def lazy_load(self) -> Iterator[Document]:
        """Transcribes the audio file and loads the transcript into documents.

        It uses the AssemblyAI API to transcribe the audio file and blocks until
        the transcription is finished.
        """
        transcript = self.transcriber.transcribe(self.file_path)
        if transcript.error:
            raise ValueError(f'Could not transcribe file: {transcript.error}')
        if self.transcript_format == TranscriptFormat.TEXT:
            yield Document(page_content=transcript.text, metadata=transcript.json_response)
        elif self.transcript_format == TranscriptFormat.SENTENCES:
            sentences = transcript.get_sentences()
            for s in sentences:
                yield Document(page_content=s.text, metadata=s.dict(exclude={'text'}))
        elif self.transcript_format == TranscriptFormat.PARAGRAPHS:
            paragraphs = transcript.get_paragraphs()
            for p in paragraphs:
                yield Document(page_content=p.text, metadata=p.dict(exclude={'text'}))
        elif self.transcript_format == TranscriptFormat.SUBTITLES_SRT:
            yield Document(page_content=transcript.export_subtitles_srt())
        elif self.transcript_format == TranscriptFormat.SUBTITLES_VTT:
            yield Document(page_content=transcript.export_subtitles_vtt())
        else:
            raise ValueError('Unknown transcript format.')