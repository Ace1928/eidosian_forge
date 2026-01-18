from unittest.mock import patch
import pytest
import torch
import os
import shutil
from llama_recipes.utils.train_utils import train
def test_save_to_json(temp_output_dir, mocker):
    model = mocker.MagicMock(name='model')
    model().loss.__truediv__().detach.return_value = torch.tensor(1)
    mock_tensor = mocker.MagicMock(name='tensor')
    batch = {'input': mock_tensor}
    train_dataloader = [batch, batch, batch, batch, batch]
    eval_dataloader = None
    tokenizer = mocker.MagicMock()
    optimizer = mocker.MagicMock()
    lr_scheduler = mocker.MagicMock()
    gradient_accumulation_steps = 1
    train_config = mocker.MagicMock()
    train_config.enable_fsdp = False
    train_config.use_fp16 = False
    train_config.run_validation = False
    train_config.gradient_clipping = False
    train_config.save_metrics = True
    train_config.max_train_step = 0
    train_config.max_eval_step = 0
    train_config.output_dir = temp_output_dir
    results = train(model, train_dataloader, eval_dataloader, tokenizer, optimizer, lr_scheduler, gradient_accumulation_steps, train_config, local_rank=0)
    assert results['metrics_filename'] not in ['', None]
    assert os.path.isfile(results['metrics_filename'])