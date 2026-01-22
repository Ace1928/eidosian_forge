import asyncio
import time
from functools import partial
from typing import (Any, Dict, Iterable, List, Optional, Set, Tuple, Type,
from vllm.lora.request import LoRARequest
from vllm.config import ModelConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm.engine.ray_utils import initialize_cluster, ray
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
class _AsyncLLMEngine(LLMEngine):
    """Extension of LLMEngine to add async methods."""

    async def step_async(self) -> List[RequestOutput]:
        """Performs one decoding iteration and returns newly generated results.
        The workers are ran asynchronously if possible.

        This function performs one decoding iteration of the engine. It first
        schedules the sequences to be executed in the next iteration and the
        token blocks to be swapped in/out/copy. Then, it executes the model
        and updates the scheduler with the model outputs. Finally, it decodes
        the sequences and returns the newly generated results.
        """
        seq_group_metadata_list, scheduler_outputs = self.scheduler.schedule()
        if not scheduler_outputs.is_empty():
            all_outputs = await self._run_workers_async('execute_model', driver_kwargs={'seq_group_metadata_list': seq_group_metadata_list, 'blocks_to_swap_in': scheduler_outputs.blocks_to_swap_in, 'blocks_to_swap_out': scheduler_outputs.blocks_to_swap_out, 'blocks_to_copy': scheduler_outputs.blocks_to_copy})
            output = all_outputs[0]
        else:
            output = []
        return self._process_model_outputs(output, scheduler_outputs)

    async def encode_request_async(self, request_id: str, prompt: Optional[str], prompt_token_ids: Optional[List[int]]=None, lora_request: Optional[LoRARequest]=None):
        if prompt_token_ids is None:
            assert prompt is not None
            prompt_token_ids = await self.tokenizer.encode_async(request_id=request_id, prompt=prompt, lora_request=lora_request)
        return prompt_token_ids

    async def add_request_async(self, request_id: str, prompt: Optional[str], sampling_params: SamplingParams, prompt_token_ids: Optional[List[int]]=None, arrival_time: Optional[float]=None, lora_request: Optional[LoRARequest]=None, prefix_pos: Optional[int]=None) -> None:
        if lora_request is not None and (not self.lora_config):
            raise ValueError(f'Got lora_request {lora_request} but LoRA is not enabled!')
        if arrival_time is None:
            arrival_time = time.time()
        prompt_token_ids = await self.encode_request_async(request_id=request_id, prompt=prompt, prompt_token_ids=prompt_token_ids, lora_request=lora_request)
        return self.add_request(request_id, prompt=prompt, prompt_token_ids=prompt_token_ids, sampling_params=sampling_params, arrival_time=arrival_time, lora_request=lora_request, prefix_pos=prefix_pos)

    async def _run_workers_async(self, method: str, *args, driver_args: Optional[List[Any]]=None, driver_kwargs: Optional[Dict[str, Any]]=None, **kwargs) -> Any:
        """Runs the given method on all workers."""
        coros = []
        if driver_args is None:
            driver_args = args
        if driver_kwargs is None:
            driver_kwargs = kwargs
        driver_executor = getattr(self.driver_worker, method)
        coros.append(asyncio.get_event_loop().run_in_executor(None, partial(driver_executor, *driver_args, **driver_kwargs)))
        for worker in self.workers:
            coros.append(worker.execute_method.remote(method, *args, **kwargs))
        all_outputs = await asyncio.gather(*coros)
        return all_outputs