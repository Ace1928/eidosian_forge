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
def test_sft_trainer_with_model_num_train_epochs(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
        training_args = TrainingArguments(output_dir=tmp_dir, dataloader_drop_last=True, evaluation_strategy='steps', max_steps=2, eval_steps=1, save_steps=1, num_train_epochs=2, per_device_train_batch_size=2)
        trainer = SFTTrainer(model=self.model, args=training_args, train_dataset=self.train_dataset, eval_dataset=self.eval_dataset, packing=True)
        trainer.train()
        assert trainer.state.log_history[-1]['train_loss'] is not None
        assert trainer.state.log_history[0]['eval_loss'] is not None
        assert 'model.safetensors' in os.listdir(tmp_dir + '/checkpoint-2')
    with tempfile.TemporaryDirectory() as tmp_dir:
        training_args = TrainingArguments(output_dir=tmp_dir, dataloader_drop_last=True, evaluation_strategy='steps', max_steps=2, save_steps=1, num_train_epochs=2, per_device_train_batch_size=2)
        trainer = SFTTrainer(model=self.model, args=training_args, train_dataset=self.dummy_dataset, dataset_text_field='text', max_seq_length=16, num_of_sequences=16, packing=True)
        trainer.train()
        assert trainer.state.log_history[-1]['train_loss'] is not None
        assert 'model.safetensors' in os.listdir(tmp_dir + '/checkpoint-2')
    with tempfile.TemporaryDirectory() as tmp_dir:
        training_args = TrainingArguments(output_dir=tmp_dir, dataloader_drop_last=True, evaluation_strategy='steps', max_steps=2, save_steps=1, num_train_epochs=2, per_device_train_batch_size=2)
        trainer = SFTTrainer(model=self.model, args=training_args, train_dataset=self.dummy_dataset, dataset_text_field='text', max_seq_length=16)
        trainer.train()
        assert trainer.state.log_history[-1]['train_loss'] is not None
        assert 'model.safetensors' in os.listdir(tmp_dir + '/checkpoint-1')