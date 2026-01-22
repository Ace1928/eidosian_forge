import asyncio
from dataclasses import dataclass
from http import HTTPStatus
from typing import Dict, List, Optional, Union
from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer import get_tokenizer
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.protocol import (CompletionRequest,
from vllm.lora.request import LoRARequest
@dataclass
class LoRA:
    name: str
    local_path: str