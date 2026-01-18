import gc
import itertools
import tempfile
import unittest
import torch
from accelerate.utils.memory import release_memory
from datasets import load_dataset
from parameterized import parameterized
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer, is_peft_available
from trl.models.utils import setup_chat_format
from ..testing_utils import require_bitsandbytes, require_peft, require_torch_gpu, require_torch_multi_gpu
from .testing_constants import DEVICE_MAP_OPTIONS, GRADIENT_CHECKPOINTING_KWARGS, MODELS_TO_TEST, PACKING_OPTIONS
@parameterized.expand(list(itertools.product(MODELS_TO_TEST, PACKING_OPTIONS, GRADIENT_CHECKPOINTING_KWARGS)))
def test_sft_trainer_transformers_mp_gc(self, model_name, packing, gradient_checkpointing_kwargs):
    """
        Simply tests if passing a transformers model to `SFTTrainer` loads and runs the trainer
        as expected in mixed precision + different scenarios of gradient_checkpointing.
        """
    with tempfile.TemporaryDirectory() as tmp_dir:
        args = TrainingArguments(output_dir=tmp_dir, logging_strategy='no', report_to='none', per_device_train_batch_size=2, max_steps=10, fp16=True, gradient_checkpointing=True, gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        trainer = SFTTrainer(model, args=args, tokenizer=tokenizer, train_dataset=self.train_dataset, eval_dataset=self.eval_dataset, packing=packing, dataset_text_field=self.dataset_text_field, max_seq_length=self.max_seq_length)
        trainer.train()
    release_memory(model, trainer)