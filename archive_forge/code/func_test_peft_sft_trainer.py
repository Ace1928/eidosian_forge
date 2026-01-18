import copy
import os
import tempfile
import unittest
import numpy as np
import pytest
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from trl.import_utils import is_peft_available
from trl.trainer import ConstantLengthDataset, DataCollatorForCompletionOnlyLM
from .testing_utils import require_peft
@require_peft
def test_peft_sft_trainer(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
        training_args = TrainingArguments(output_dir=tmp_dir, dataloader_drop_last=True, evaluation_strategy='steps', max_steps=4, eval_steps=2, save_steps=2, per_device_train_batch_size=2)
        peft_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias='none', task_type='CAUSAL_LM')
        trainer = SFTTrainer(model=self.model_id, args=training_args, train_dataset=self.train_dataset, eval_dataset=self.eval_dataset, peft_config=peft_config, packing=True)
        assert isinstance(trainer.model, PeftModel)
        trainer.train()
        assert trainer.state.log_history[-1]['train_loss'] is not None
        assert trainer.state.log_history[0]['eval_loss'] is not None
        assert 'adapter_model.safetensors' in os.listdir(tmp_dir + '/checkpoint-2')
        assert 'adapter_config.json' in os.listdir(tmp_dir + '/checkpoint-2')
        assert 'model.safetensors' not in os.listdir(tmp_dir + '/checkpoint-2')