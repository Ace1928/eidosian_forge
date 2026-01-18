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
def test_ppo_step_rewards_shape(self):
    """
        Test if the rewards shape is correct by asserting that if a wrong reward shape is passed, we get
        a value error.
        """
    dummy_dataset = self._init_dummy_dataset()
    ppo_trainer = PPOTrainer(config=self.ppo_config, model=self.gpt2_model, ref_model=None, tokenizer=self.gpt2_tokenizer, dataset=dummy_dataset)
    dummy_dataloader = ppo_trainer.dataloader
    for query_tensor, response_tensor in dummy_dataloader:
        reward = [torch.tensor([[1.0]]), torch.tensor([[0.0]])]
        with pytest.raises(ValueError):
            _ = ppo_trainer.step(list(query_tensor), list(response_tensor), reward)
        reward = [torch.tensor([1.0]), torch.tensor([0.0])]
        _ = ppo_trainer.step(list(query_tensor), list(response_tensor), reward)
        break
    for name, param in ppo_trainer.model.named_parameters():
        assert param.grad is not None, f'Parameter {name} has no gradient'
    for name, param in ppo_trainer.ref_model.named_parameters():
        assert param.grad is None, f'Parameter {name} has a gradient'