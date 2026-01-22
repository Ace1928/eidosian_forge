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
def _load_chat_template(self, chat_template):
    if chat_template is not None:
        try:
            with open(chat_template, 'r') as f:
                self.tokenizer.chat_template = f.read()
        except OSError:
            self.tokenizer.chat_template = codecs.decode(chat_template, 'unicode_escape')
        logger.info(f'Using supplied chat template:\n{self.tokenizer.chat_template}')
    elif self.tokenizer.chat_template is not None:
        logger.info(f'Using default chat template:\n{self.tokenizer.chat_template}')
    else:
        logger.warning('No chat template provided. Chat API will not work.')