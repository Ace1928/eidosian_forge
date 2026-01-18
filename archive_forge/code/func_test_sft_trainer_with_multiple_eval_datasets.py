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
def test_sft_trainer_with_multiple_eval_datasets(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
        training_args = TrainingArguments(output_dir=tmp_dir, dataloader_drop_last=True, evaluation_strategy='steps', max_steps=1, eval_steps=1, save_steps=1, per_device_train_batch_size=2)
        trainer = SFTTrainer(model=self.model_id, args=training_args, train_dataset=self.train_dataset, eval_dataset={'data1': self.eval_dataset, 'data2': self.eval_dataset}, packing=True)
        trainer.train()
        assert trainer.state.log_history[-1]['train_loss'] is not None
        assert trainer.state.log_history[0]['eval_data1_loss'] is not None
        assert trainer.state.log_history[1]['eval_data2_loss'] is not None
        assert 'model.safetensors' in os.listdir(tmp_dir + '/checkpoint-1')