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
def test_ppo_step_with_no_ref_sgd(self):
    dummy_dataset = self._init_dummy_dataset()
    optimizer = torch.optim.SGD(self.gpt2_model.parameters(), lr=0.01)
    ppo_trainer = PPOTrainer(config=self.ppo_config, model=self.gpt2_model, ref_model=None, optimizer=optimizer, tokenizer=self.gpt2_tokenizer, dataset=dummy_dataset)
    dummy_dataloader = ppo_trainer.dataloader
    assert isinstance(ppo_trainer.optimizer.optimizer, torch.optim.SGD)
    for query_tensor, response_tensor in dummy_dataloader:
        reward = [torch.tensor(1.0), torch.tensor(0.0)]
        train_stats = ppo_trainer.step(list(query_tensor), list(response_tensor), reward)
        break
    for name, param in ppo_trainer.model.named_parameters():
        assert param.grad is not None, f'Parameter {name} has no gradient'
    for name, param in ppo_trainer.ref_model.named_parameters():
        assert param.grad is None, f'Parameter {name} has a gradient'
    for stat in EXPECTED_STATS:
        assert stat in train_stats.keys()