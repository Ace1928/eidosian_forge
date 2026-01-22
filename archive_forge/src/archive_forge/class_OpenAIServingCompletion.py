import asyncio
import time
from fastapi import Request
from typing import AsyncGenerator, AsyncIterator, Callable, List, Optional, Dict, Tuple
from vllm.logger import init_logger
from vllm.utils import random_uuid
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.protocol import (
from vllm.outputs import RequestOutput
from vllm.entrypoints.openai.serving_engine import OpenAIServing, LoRA
from vllm.model_executor.guided_decoding import get_guided_decoding_logits_processor
class OpenAIServingCompletion(OpenAIServing):

    def __init__(self, engine: AsyncLLMEngine, served_model: str, lora_modules: Optional[List[LoRA]]=None):
        super().__init__(engine=engine, served_model=served_model, lora_modules=lora_modules)

    async def create_completion(self, request: CompletionRequest, raw_request: Request):
        """Completion API similar to OpenAI's API.

        See https://platform.openai.com/docs/api-reference/completions/create
        for the API specification. This API mimics the OpenAI Completion API.

        NOTE: Currently we do not support the following feature:
            - suffix (the language models we currently support do not support
            suffix)
        """
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            return error_check_ret
        if request.suffix is not None:
            return self.create_error_response('suffix is not currently supported')
        model_name = request.model
        request_id = f'cmpl-{random_uuid()}'
        created_time = int(time.monotonic())
        generators = []
        try:
            sampling_params = request.to_sampling_params()
            lora_request = self._maybe_get_lora(request)
            guided_decode_logit_processor = await get_guided_decoding_logits_processor(request, self.engine.get_tokenizer())
            if guided_decode_logit_processor is not None:
                if sampling_params.logits_processors is None:
                    sampling_params.logits_processors = []
                sampling_params.logits_processors.append(guided_decode_logit_processor)
            prompt_is_tokens, prompts = parse_prompt_format(request.prompt)
            for i, prompt in enumerate(prompts):
                if prompt_is_tokens:
                    input_ids = self._validate_prompt_and_tokenize(request, prompt_ids=prompt)
                else:
                    input_ids = self._validate_prompt_and_tokenize(request, prompt=prompt)
                generators.append(self.engine.generate(prompt, sampling_params, f'{request_id}-{i}', prompt_token_ids=input_ids, lora_request=lora_request))
        except ValueError as e:
            return self.create_error_response(str(e))
        result_generator: AsyncIterator[Tuple[int, RequestOutput]] = merge_async_iterators(*generators)
        stream = request.stream and (request.best_of is None or request.n == request.best_of) and (not request.use_beam_search)
        if stream:
            return completion_stream_generator(request, raw_request, self.engine.abort, result_generator, self._create_logprobs, request_id, created_time, model_name, num_prompts=len(prompts))
        final_res_batch: RequestOutput = [None] * len(prompts)
        async for i, res in result_generator:
            if await raw_request.is_disconnected():
                await self.engine.abort(f'{request_id}-{i}')
                return self.create_error_response('Client disconnected')
            final_res_batch[i] = res
        response = request_output_to_completion_response(final_res_batch, request, self._create_logprobs, request_id, created_time, model_name)
        if request.stream:
            response_json = response.model_dump_json()

            async def fake_stream_generator() -> AsyncGenerator[str, None]:
                yield f'data: {response_json}\n\n'
                yield 'data: [DONE]\n\n'
            return fake_stream_generator()
        return response