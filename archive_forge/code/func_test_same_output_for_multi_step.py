import torch
import random
import pytest
from unittest.mock import MagicMock
from vllm.worker.spec_decode.multi_step_worker import MultiStepWorker
from vllm.worker.worker import Worker
from vllm.model_executor.utils import set_random_seed
from .utils import (create_execute_model_data, create_worker,
@torch.inference_mode()
def test_same_output_for_multi_step():
    """Verify the multi-step worker produces the same output as the normal
    worker when num_steps > 1. This test runs the multi-step worker once, and
    then runs the worker num_steps times, and compares the output.
    """
    seed = 100
    model_name = 'JackFram/llama-68m'
    block_size = 16
    num_gpu_blocks = 2048 // block_size
    multi_step_worker = create_worker(MultiStepWorker, model_name, block_size, num_gpu_blocks, seed)
    worker = create_worker(Worker, model_name, block_size, num_gpu_blocks, seed)
    num_steps = block_size + 1
    random.seed(seed)
    prompts = [[random.randint(0, 1000) for _ in range(random.randint(10, 20))] for _ in range(10)]
    final_seq_lens = [len(prompt) + num_steps for prompt in prompts]
    rand_seeds = list((random.randint(0, 100) for _ in range(num_steps)))
    multi_step_worker.execute_model = patch_execute_model_with_seeds(multi_step_worker, rand_seeds)
    worker.execute_model = patch_execute_model_with_seeds(worker, rand_seeds)
    continuations = [[1] for _ in prompts]
    execute_model_data = create_execute_model_data(create_seq_group_metadata_from_prompts(prompts, num_gpu_blocks, block_size, continuations=continuations, final_seq_lens=final_seq_lens))
    zero_kv_cache(multi_step_worker.cache_engine)
    set_random_seed(seed)
    multi_step_output = multi_step_worker.execute_model_multi_step(**execute_model_data.to_dict(), num_steps=num_steps)
    zero_kv_cache(worker.cache_engine)
    single_step_output = []
    continuations = [[1] for _ in prompts]
    set_random_seed(seed)
    for _ in multi_step_output:
        execute_model_data = create_execute_model_data(create_seq_group_metadata_from_prompts(prompts, num_gpu_blocks, block_size, continuations=continuations, final_seq_lens=final_seq_lens))
        single_step_output.append(worker.execute_model(**execute_model_data.to_dict()))
        for i, seq_group_output in enumerate(single_step_output[-1]):
            continuations[i].append(seq_group_output.samples[0].output_token)
    multi_step_output_logprobs = [[] for _ in prompts]
    single_step_output_logprobs = [[] for _ in prompts]
    multi_step_output_token_ids = [[] for _ in prompts]
    single_step_output_token_ids = [[] for _ in prompts]
    for i, _ in enumerate(prompts):
        for multi_step, single_step in zip(multi_step_output, single_step_output):
            multi_step_output_token_ids[i].append(multi_step[i].samples[0].output_token)
            single_step_output_token_ids[i].append(single_step[i].samples[0].output_token)
            multi_step_output_logprobs[i].append(multi_step[i].samples[0].logprobs)
            single_step_output_logprobs[i].append(single_step[i].samples[0].logprobs)
    for i, (multi_step_tokens, single_step_tokens) in enumerate(zip(multi_step_output_token_ids, single_step_output_token_ids)):
        print(f'i={i!r} multi_step_tokens={multi_step_tokens!r}')
        print(f'i={i!r} single_step_tokens={single_step_tokens!r}')
        print(f'i={i!r} equal {multi_step_tokens == single_step_tokens}')
    for multi_step_tokens, single_step_tokens in zip(multi_step_output_token_ids, single_step_output_token_ids):
        assert multi_step_tokens == single_step_tokens
    for multi_step_logprobs, single_step_logprobs in zip(multi_step_output_logprobs, single_step_output_logprobs):
        assert_logprobs_dict_allclose(multi_step_logprobs, single_step_logprobs)