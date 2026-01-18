import copy
import fnmatch
import gc
import re
import tempfile
import unittest
import pytest
import torch
from huggingface_hub import HfApi, HfFolder, delete_repo
from parameterized import parameterized
from pytest import mark
from requests.exceptions import HTTPError
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, AutoModelForSeq2SeqLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import respond_to_batch
from .testing_constants import CI_HUB_ENDPOINT, CI_HUB_USER, CI_HUB_USER_TOKEN
from .testing_utils import require_peft, require_torch_multi_gpu
def test_ppo_trainer_full_kl_penalty(self):
    dummy_dataset = self._init_dummy_dataset()
    self.ppo_config.kl_penalty = 'full'
    ppo_trainer = PPOTrainer(config=self.ppo_config, model=self.gpt2_model, ref_model=None, tokenizer=self.gpt2_tokenizer, dataset=dummy_dataset)
    log_probs = torch.Tensor([[[0.1, 0.2, 0.7], [0.3, 0.4, 0.3]]]).exp()
    ref_log_probs = torch.Tensor([[[0.1, 0.2, 0.7], [0.3, 0.4, 0.3]]]).exp()
    expected_output = torch.Tensor([[0.0, 0.0]])
    output = ppo_trainer._kl_penalty(log_probs, ref_log_probs)
    assert output.shape == (1, 2)
    assert torch.allclose(output, expected_output)
    log_probs = torch.Tensor([[[0.98, 0.01, 0.01], [0.01, 0.98, 0.01]]]).log()
    ref_log_probs = torch.Tensor([[[0.01, 0.01, 0.98], [0.01, 0.01, 0.98]]]).log()
    expected_output = torch.Tensor([[4.4474, 4.4474]])
    output = ppo_trainer._kl_penalty(log_probs, ref_log_probs)
    assert output.shape == (1, 2)
    assert torch.allclose(output, expected_output)
    log_probs = torch.Tensor([[[0.49, 0.02, 0.49], [0.49, 0.02, 0.49]]]).log()
    ref_log_probs = torch.Tensor([[[0.01, 0.98, 0.01], [0.49, 0.02, 0.49]]]).log()
    expected_output = torch.Tensor([[3.7361, 0.0]])
    output = ppo_trainer._kl_penalty(log_probs, ref_log_probs)
    assert output.shape == (1, 2)
    assert torch.allclose(output, expected_output, atol=0.0001)