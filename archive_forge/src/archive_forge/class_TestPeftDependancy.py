import sys
import unittest
from unittest.mock import patch
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .testing_utils import is_peft_available, require_peft
@require_peft
class TestPeftDependancy(unittest.TestCase):

    def setUp(self):
        self.causal_lm_model_id = 'trl-internal-testing/tiny-random-GPTNeoXForCausalLM'
        self.seq_to_seq_model_id = 'trl-internal-testing/tiny-random-T5ForConditionalGeneration'
        if is_peft_available():
            from peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias='none', task_type='CAUSAL_LM')
            causal_lm_model = AutoModelForCausalLM.from_pretrained(self.causal_lm_model_id)
            self.peft_model = get_peft_model(causal_lm_model, lora_config)

    def test_no_peft(self):
        with patch.dict(sys.modules, {'peft': None}):
            from trl import AutoModelForCausalLMWithValueHead, AutoModelForSeq2SeqLMWithValueHead
            with pytest.raises(ModuleNotFoundError):
                import peft
            _trl_model = AutoModelForCausalLMWithValueHead.from_pretrained(self.causal_lm_model_id)
            _trl_seq2seq_model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(self.seq_to_seq_model_id)

    def test_imports_no_peft(self):
        with patch.dict(sys.modules, {'peft': None}):
            from trl import AutoModelForCausalLMWithValueHead, AutoModelForSeq2SeqLMWithValueHead, PPOConfig, PPOTrainer, PreTrainedModelWrapper

    def test_ppo_trainer_no_peft(self):
        with patch.dict(sys.modules, {'peft': None}):
            from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
            ppo_model_id = 'trl-internal-testing/dummy-GPT2-correct-vocab'
            trl_model = AutoModelForCausalLMWithValueHead.from_pretrained(ppo_model_id)
            tokenizer = AutoTokenizer.from_pretrained(ppo_model_id)
            tokenizer.pad_token_id = tokenizer.eos_token_id
            ppo_config = PPOConfig(batch_size=2, mini_batch_size=1, log_with=None)
            dummy_dataset = DummyDataset([torch.LongTensor([0, 1, 0, 1, 0, 1]), torch.LongTensor([0, 1, 0, 1, 0, 1])], [torch.LongTensor([1, 0, 1, 0, 1, 0]), torch.LongTensor([0, 1, 0, 1, 0, 1])])
            ppo_trainer = PPOTrainer(config=ppo_config, model=trl_model, ref_model=None, tokenizer=tokenizer, dataset=dummy_dataset)
            dummy_dataloader = ppo_trainer.dataloader
            for query_tensor, response_tensor in dummy_dataloader:
                reward = [torch.tensor(1.0), torch.tensor(0.0)]
                train_stats = ppo_trainer.step(list(query_tensor), list(response_tensor), reward)
                break
            for _, param in trl_model.named_parameters():
                if param.requires_grad:
                    assert param.grad is not None
            for stat in EXPECTED_STATS:
                assert stat in train_stats