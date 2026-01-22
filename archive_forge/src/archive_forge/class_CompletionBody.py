import asyncio
import json
from threading import Lock
from typing import List, Union
from enum import Enum
import base64
from fastapi import APIRouter, Request, status, HTTPException
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel, Field
import tiktoken
from utils.rwkv import *
from utils.log import quick_log
import global_var
class CompletionBody(ModelConfigBody):
    prompt: Union[str, List[str], None]
    model: Union[str, None] = 'rwkv'
    stream: bool = False
    stop: Union[str, List[str], None] = None
    model_config = {'json_schema_extra': {'example': {'prompt': 'The following is an epic science fiction masterpiece that is immortalized, ' + 'with delicate descriptions and grand depictions of interstellar civilization wars.\nChapter 1.\n', 'model': 'rwkv', 'stream': False, 'stop': None, 'max_tokens': 100, 'temperature': 1, 'top_p': 0.3, 'presence_penalty': 0, 'frequency_penalty': 1}}}