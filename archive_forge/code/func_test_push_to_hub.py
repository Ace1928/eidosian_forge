import gc
import tempfile
import unittest
import pytest
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from trl import AutoModelForCausalLMWithValueHead, AutoModelForSeq2SeqLMWithValueHead, create_reference_model
@unittest.skip('This test needs to be run manually due to HF token issue.')
def test_push_to_hub(self):
    for model_name in self.all_model_names:
        model = self.trl_model_class.from_pretrained(model_name)
        if 'sharded' in model_name:
            model.push_to_hub(model_name + '-ppo', use_auth_token=True, max_shard_size='1MB')
        else:
            model.push_to_hub(model_name + '-ppo', use_auth_token=True)
        model_from_pretrained = self.trl_model_class.from_pretrained(model_name + '-ppo')
        assert model.state_dict().keys() == model_from_pretrained.state_dict().keys()
        for name, param in model.state_dict().items():
            assert torch.allclose(param, model_from_pretrained.state_dict()[name]), f'Parameter {name} is not the same after push_to_hub and from_pretrained'