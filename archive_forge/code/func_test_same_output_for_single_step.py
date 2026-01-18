import torch
import random
import pytest
from unittest.mock import MagicMock
from vllm.worker.spec_decode.multi_step_worker import MultiStepWorker
from vllm.worker.worker import Worker
from vllm.model_executor.utils import set_random_seed
from .utils import (create_execute_model_data, create_worker,
@torch.inference_mode()
def test_same_output_for_single_step():
    """Verify the multi step worker produces the same output as the normal
    worker for num_steps=1.
    """
    seed = 100
    model_name = 'JackFram/llama-68m'
    block_size = 32
    num_gpu_blocks = 2048 // block_size
    multi_step_worker = create_worker(MultiStepWorker, model_name, block_size, num_gpu_blocks, seed)
    worker = create_worker(Worker, model_name, block_size, num_gpu_blocks, seed)
    multi_step_worker.model_runner = worker.model_runner
    multi_step_worker.cache_engine = worker.cache_engine
    num_steps = 1
    prompts = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
    final_seq_lens = [len(prompt) + num_steps for prompt in prompts]
    multi_step_execute_model_data = create_execute_model_data(seq_group_metadata_list=create_seq_group_metadata_from_prompts(prompts, num_gpu_blocks, block_size, final_seq_lens=final_seq_lens))
    single_step_execute_model_data = create_execute_model_data(seq_group_metadata_list=create_seq_group_metadata_from_prompts(prompts, num_gpu_blocks, block_size, final_seq_lens=final_seq_lens))
    zero_kv_cache(multi_step_worker.cache_engine)
    set_random_seed(seed)
    actual_output = multi_step_worker.execute_model_multi_step(**multi_step_execute_model_data.to_dict(), num_steps=num_steps)
    assert len(actual_output) == num_steps
    actual_output = actual_output[0]
    zero_kv_cache(worker.cache_engine)
    set_random_seed(seed)
    expected_output = worker.execute_model(**single_step_execute_model_data.to_dict())
    actual_token_ids = [output.samples[0].output_token for output in actual_output]
    actual_logprobs = [output.samples[0].logprobs for output in actual_output]
    expected_token_ids = [output.samples[0].output_token for output in expected_output]
    expected_logprobs = [output.samples[0].logprobs for output in expected_output]
    assert actual_token_ids == expected_token_ids
    print(f'actual_logprobs={actual_logprobs!r}')
    print(f'expected_logprobs={expected_logprobs!r}')
    assert_logprobs_dict_allclose(actual_logprobs, expected_logprobs)