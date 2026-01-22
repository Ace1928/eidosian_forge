from __future__ import annotations
import asyncio
import functools
import hashlib
import hmac
import json
import os
import re
import shutil
import sys
from collections import deque
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass as python_dataclass
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile, _TemporaryFileWrapper
from typing import (
from urllib.parse import urlparse
import anyio
import fastapi
import gradio_client.utils as client_utils
import httpx
import multipart
from gradio_client.documentation import document
from multipart.multipart import parse_options_header
from starlette.datastructures import FormData, Headers, MutableHeaders, UploadFile
from starlette.formparsers import MultiPartException, MultipartPart
from starlette.responses import PlainTextResponse, Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send
from gradio import processing_utils, utils
from gradio.data_classes import PredictBody
from gradio.exceptions import Error
from gradio.helpers import EventData
from gradio.state_holder import SessionState
class GradioMultiPartParser:
    """Vendored from starlette.MultipartParser.

    Thanks starlette!

    Made the following modifications
        - Use GradioUploadFile instead of UploadFile
        - Use NamedTemporaryFile instead of SpooledTemporaryFile
        - Compute hash of data as the request is streamed

    """
    max_file_size = 1024 * 1024

    def __init__(self, headers: Headers, stream: AsyncGenerator[bytes, None], *, max_files: Union[int, float]=1000, max_fields: Union[int, float]=1000, upload_id: str | None=None, upload_progress: FileUploadProgress | None=None) -> None:
        self.headers = headers
        self.stream = stream
        self.max_files = max_files
        self.max_fields = max_fields
        self.items: List[Tuple[str, Union[str, UploadFile]]] = []
        self.upload_id = upload_id
        self.upload_progress = upload_progress
        self._current_files = 0
        self._current_fields = 0
        self._current_partial_header_name: bytes = b''
        self._current_partial_header_value: bytes = b''
        self._current_part = MultipartPart()
        self._charset = ''
        self._file_parts_to_write: List[Tuple[MultipartPart, bytes]] = []
        self._file_parts_to_finish: List[MultipartPart] = []
        self._files_to_close_on_error: List[_TemporaryFileWrapper] = []

    def on_part_begin(self) -> None:
        self._current_part = MultipartPart()

    def on_part_data(self, data: bytes, start: int, end: int) -> None:
        message_bytes = data[start:end]
        if self.upload_progress is not None:
            self.upload_progress.append(self.upload_id, self._current_part.file.filename, message_bytes)
        if self._current_part.file is None:
            self._current_part.data += message_bytes
        else:
            self._file_parts_to_write.append((self._current_part, message_bytes))

    def on_part_end(self) -> None:
        if self._current_part.file is None:
            self.items.append((self._current_part.field_name, _user_safe_decode(self._current_part.data, self._charset)))
        else:
            self._file_parts_to_finish.append(self._current_part)
            self.items.append((self._current_part.field_name, self._current_part.file))

    def on_header_field(self, data: bytes, start: int, end: int) -> None:
        self._current_partial_header_name += data[start:end]

    def on_header_value(self, data: bytes, start: int, end: int) -> None:
        self._current_partial_header_value += data[start:end]

    def on_header_end(self) -> None:
        field = self._current_partial_header_name.lower()
        if field == b'content-disposition':
            self._current_part.content_disposition = self._current_partial_header_value
        self._current_part.item_headers.append((field, self._current_partial_header_value))
        self._current_partial_header_name = b''
        self._current_partial_header_value = b''

    def on_headers_finished(self) -> None:
        _, options = parse_options_header(self._current_part.content_disposition or b'')
        try:
            self._current_part.field_name = _user_safe_decode(options[b'name'], str(self._charset))
        except KeyError as e:
            raise MultiPartException('The Content-Disposition header field "name" must be provided.') from e
        if b'filename' in options:
            self._current_files += 1
            if self._current_files > self.max_files:
                raise MultiPartException(f'Too many files. Maximum number of files is {self.max_files}.')
            filename = _user_safe_decode(options[b'filename'], str(self._charset))
            tempfile = NamedTemporaryFile(delete=False)
            self._files_to_close_on_error.append(tempfile)
            self._current_part.file = GradioUploadFile(file=tempfile, size=0, filename=filename, headers=Headers(raw=self._current_part.item_headers))
        else:
            self._current_fields += 1
            if self._current_fields > self.max_fields:
                raise MultiPartException(f'Too many fields. Maximum number of fields is {self.max_fields}.')
            self._current_part.file = None

    def on_end(self) -> None:
        pass

    async def parse(self) -> FormData:
        _, params = parse_options_header(self.headers['Content-Type'])
        charset = params.get(b'charset', 'utf-8')
        if isinstance(charset, bytes):
            charset = charset.decode('latin-1')
        self._charset = charset
        try:
            boundary = params[b'boundary']
        except KeyError as e:
            raise MultiPartException('Missing boundary in multipart.') from e
        callbacks: multipart.multipart.MultipartCallbacks = {'on_part_begin': self.on_part_begin, 'on_part_data': self.on_part_data, 'on_part_end': self.on_part_end, 'on_header_field': self.on_header_field, 'on_header_value': self.on_header_value, 'on_header_end': self.on_header_end, 'on_headers_finished': self.on_headers_finished, 'on_end': self.on_end}
        parser = multipart.MultipartParser(boundary, callbacks)
        try:
            async for chunk in self.stream:
                parser.write(chunk)
                for part, data in self._file_parts_to_write:
                    assert part.file
                    await part.file.write(data)
                    part.file.sha.update(data)
                for part in self._file_parts_to_finish:
                    assert part.file
                    await part.file.seek(0)
                self._file_parts_to_write.clear()
                self._file_parts_to_finish.clear()
        except MultiPartException as exc:
            for file in self._files_to_close_on_error:
                file.close()
            raise exc
        parser.finalize()
        if self.upload_progress is not None:
            self.upload_progress.set_done(self.upload_id)
        return FormData(self.items)