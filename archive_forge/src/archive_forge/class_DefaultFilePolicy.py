import base64
import functools
import itertools
import logging
import os
import queue
import random
import sys
import threading
import time
from types import TracebackType
from typing import (
import requests
import wandb
from wandb import util
from wandb.sdk.internal import internal_api
from ..lib import file_stream_utils
class DefaultFilePolicy:

    def __init__(self, start_chunk_id: int=0) -> None:
        self._chunk_id = start_chunk_id

    def process_chunks(self, chunks: List[Chunk]) -> Union[bool, 'ProcessedChunk', 'ProcessedBinaryChunk', List['ProcessedChunk']]:
        chunk_id = self._chunk_id
        self._chunk_id += len(chunks)
        return {'offset': chunk_id, 'content': [c.data for c in chunks]}