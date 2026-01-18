import asyncio
import concurrent.futures
from copy import copy
from enum import Enum
from functools import lru_cache
from json import dumps as json_dumps
from re import escape as regex_escape
from typing import Union, Tuple
from pydantic import BaseModel
from vllm.entrypoints.openai.protocol import CompletionRequest, ChatCompletionRequest
from vllm.model_executor.guided_logits_processors import JSONLogitsProcessor, RegexLogitsProcessor

    Given an OpenAI-compatible request, check for guided decoding parameters
    and get the necessary logits processor for the given guide.
    We cache logit processors by (guide, tokenizer), and on cache hit
    we make a shallow copy to reuse the same underlying FSM.
    