import time
import codecs
from fastapi import Request
from typing import AsyncGenerator, AsyncIterator, Optional, List, Union
from vllm.logger import init_logger
from vllm.utils import random_uuid
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.protocol import (
from vllm.outputs import RequestOutput
from vllm.entrypoints.openai.serving_engine import OpenAIServing, LoRA
from vllm.model_executor.guided_decoding import get_guided_decoding_logits_processor
Completion API similar to OpenAI's API.

        See  https://platform.openai.com/docs/api-reference/chat/create
        for the API specification. This API mimics the OpenAI ChatCompletion API.

        NOTE: Currently we do not support the following feature:
            - function_call (Users should implement this by themselves)
        