import tempfile
import unittest
import pytest
import torch
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction
from trl import RewardConfig, RewardTrainer
from trl.trainer import compute_accuracy
from .testing_utils import require_peft
@require_peft
def test_reward_trainer_peft(self):
    import peft
    from peft import LoraConfig, TaskType
    peft_version = peft.__version__
    peft_config = LoraConfig(task_type=TaskType.SEQ_CLS, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
    with tempfile.TemporaryDirectory() as tmp_dir:
        training_args = RewardConfig(output_dir=tmp_dir, per_device_train_batch_size=2, max_steps=6, remove_unused_columns=False, gradient_accumulation_steps=2, learning_rate=0.9, evaluation_strategy='steps')
        dummy_dataset_dict = {'input_ids_chosen': [torch.LongTensor([0, 1, 2]), torch.LongTensor([1, 2]), torch.LongTensor([0, 1, 2]), torch.LongTensor([1, 2])], 'attention_mask_chosen': [torch.LongTensor([1, 1, 1]), torch.LongTensor([1, 0]), torch.LongTensor([1, 1, 1]), torch.LongTensor([1, 0])], 'input_ids_rejected': [torch.LongTensor([0, 2]), torch.LongTensor([1, 2, 0]), torch.LongTensor([0, 2]), torch.LongTensor([1, 2, 0])], 'attention_mask_rejected': [torch.LongTensor([1, 1]), torch.LongTensor([1, 1, 0]), torch.LongTensor([1, 1]), torch.LongTensor([1, 1, 1])]}
        dummy_dataset = Dataset.from_dict(dummy_dataset_dict)
        trainer = RewardTrainer(model=self.model, args=training_args, tokenizer=self.tokenizer, train_dataset=dummy_dataset, eval_dataset=dummy_dataset, peft_config=peft_config)
        previous_trainable_params = {}
        previous_non_trainable_params = {}
        trainable_params_name = ['lora', 'score'] if peft_version < '0.3.0' else ['lora', 'modules_to_save']
        for n, param in trainer.model.named_parameters():
            if any((t in n for t in trainable_params_name)):
                previous_trainable_params[n] = param.clone()
            else:
                previous_non_trainable_params[n] = param.clone()
        trainer.train()
        assert trainer.state.log_history[-1]['train_loss'] is not None
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.allclose(param, new_param, atol=1e-12, rtol=1e-12)
        for n, param in previous_non_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert torch.allclose(param, new_param, atol=1e-12, rtol=1e-12)
        preds = trainer.predict(dummy_dataset)
        assert preds.predictions.shape == (4, 2)