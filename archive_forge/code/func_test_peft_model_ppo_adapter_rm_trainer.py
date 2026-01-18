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
@require_peft
@mark.peft_test
def test_peft_model_ppo_adapter_rm_trainer(self):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification
    dummy_inputs = torch.LongTensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
    rm_lora_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias='none', task_type='SEQ_CLS')
    reward_model = AutoModelForSequenceClassification.from_pretrained(self.model_id)
    reward_model = get_peft_model(reward_model, rm_lora_config)
    dummy_optim = torch.optim.Adam(filter(lambda p: p.requires_grad, reward_model.parameters()), lr=0.001)
    previous_rm_logits = reward_model(dummy_inputs).logits
    loss = previous_rm_logits.mean()
    loss.backward()
    dummy_optim.step()
    reward_model.eval()
    original_rm_logits = reward_model(dummy_inputs).logits
    with tempfile.TemporaryDirectory() as tmpdirname:
        reward_model.save_pretrained(tmpdirname)
        lora_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias='none', task_type='CAUSAL_LM')
        gpt2_model = AutoModelForCausalLM.from_pretrained(self.model_id)

        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        gpt2_model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        peft_model = get_peft_model(gpt2_model, lora_config)
        model = AutoModelForCausalLMWithValueHead.from_pretrained(peft_model, reward_adapter=tmpdirname)
        dummy_dataset = self._init_dummy_dataset()
        self.ppo_config.batch_size = 2
        self.ppo_config.mini_batch_size = 1
        ppo_trainer = PPOTrainer(config=self.ppo_config, model=model, ref_model=None, tokenizer=self.gpt2_tokenizer, dataset=dummy_dataset)
        assert ppo_trainer.ref_model is None
        dummy_dataloader = ppo_trainer.dataloader
        for query_tensor, response_tensor in dummy_dataloader:
            reward = [torch.tensor(1.0), torch.tensor(0.0)]
            _ = ppo_trainer.step(list(query_tensor), list(response_tensor), reward)
            ppo_trainer.model.train()
            ppo_trainer.model.gradient_checkpointing_enable()
            _ = ppo_trainer.step(list(query_tensor), list(response_tensor), reward)
            break
        new_logits = ppo_trainer.model.compute_reward_score(dummy_inputs)
        assert not torch.allclose(previous_rm_logits, new_logits[:, -1, :])
        assert torch.allclose(original_rm_logits, new_logits[:, -1, :])
        for name, param in model.named_parameters():
            if ('lora' in name or 'v_head' in name) and 'reward' not in name:
                assert param.grad is not None, f'Parameter {name} has a no gradient'
            else:
                assert param.grad is None, f'Parameter {name} has a gradient'