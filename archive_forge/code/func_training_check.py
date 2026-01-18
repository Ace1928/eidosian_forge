import contextlib
import io
import math
import time
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from accelerate import Accelerator
from accelerate.data_loader import SeedableRandomSampler, prepare_data_loader
from accelerate.state import AcceleratorState
from accelerate.test_utils import RegressionDataset, are_the_same_tensors
from accelerate.utils import (
def training_check(use_seedable_sampler=False):
    state = AcceleratorState()
    generator = torch.Generator()
    batch_size = 8
    length = batch_size * 4 * state.num_processes
    train_set, old_model = mock_training(length, batch_size * state.num_processes, generator, use_seedable_sampler)
    assert are_the_same_tensors(old_model.a), 'Did not obtain the same model on both processes.'
    assert are_the_same_tensors(old_model.b), 'Did not obtain the same model on both processes.'
    accelerator = Accelerator()
    train_dl = generate_baseline_dataloader(train_set, generator, batch_size, use_seedable_sampler)
    model = RegressionModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    train_dl, model, optimizer = accelerator.prepare(train_dl, model, optimizer)
    set_seed(42)
    generator.manual_seed(42)
    for _ in range(3):
        for batch in train_dl:
            model.zero_grad()
            output = model(batch['x'])
            loss = torch.nn.functional.mse_loss(output, batch['y'])
            accelerator.backward(loss)
            optimizer.step()
    model = accelerator.unwrap_model(model).cpu()
    assert torch.allclose(old_model.a, model.a), 'Did not obtain the same model on CPU or distributed training.'
    assert torch.allclose(old_model.b, model.b), 'Did not obtain the same model on CPU or distributed training.'
    accelerator.print('Training yielded the same results on one CPU or distributed setup with no batch split.')
    dataloader_config = DataLoaderConfiguration(split_batches=True, use_seedable_sampler=use_seedable_sampler)
    accelerator = Accelerator(dataloader_config=dataloader_config)
    train_dl = generate_baseline_dataloader(train_set, generator, batch_size * state.num_processes, use_seedable_sampler)
    model = RegressionModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    train_dl, model, optimizer = accelerator.prepare(train_dl, model, optimizer)
    set_seed(42)
    generator.manual_seed(42)
    for _ in range(3):
        for batch in train_dl:
            model.zero_grad()
            output = model(batch['x'])
            loss = torch.nn.functional.mse_loss(output, batch['y'])
            accelerator.backward(loss)
            optimizer.step()
    model = accelerator.unwrap_model(model).cpu()
    assert torch.allclose(old_model.a, model.a), 'Did not obtain the same model on CPU or distributed training.'
    assert torch.allclose(old_model.b, model.b), 'Did not obtain the same model on CPU or distributed training.'
    accelerator.print('Training yielded the same results on one CPU or distributes setup with batch split.')
    if torch.cuda.is_available() or is_npu_available():
        print('FP16 training check.')
        AcceleratorState._reset_state()
        dataloader_config = DataLoaderConfiguration(use_seedable_sampler=use_seedable_sampler)
        accelerator = Accelerator(mixed_precision='fp16', dataloader_config=dataloader_config)
        train_dl = generate_baseline_dataloader(train_set, generator, batch_size, use_seedable_sampler)
        model = RegressionModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        train_dl, model, optimizer = accelerator.prepare(train_dl, model, optimizer)
        set_seed(42)
        generator.manual_seed(42)
        for _ in range(3):
            for batch in train_dl:
                model.zero_grad()
                output = model(batch['x'])
                loss = torch.nn.functional.mse_loss(output, batch['y'])
                accelerator.backward(loss)
                optimizer.step()
        model = accelerator.unwrap_model(model).cpu()
        assert torch.allclose(old_model.a, model.a), 'Did not obtain the same model on CPU or distributed training.'
        assert torch.allclose(old_model.b, model.b), 'Did not obtain the same model on CPU or distributed training.'
    if torch.cuda.is_available():
        print('Keep fp32 wrapper check.')
        AcceleratorState._reset_state()
        accelerator = Accelerator(mixed_precision='fp16')
        model = torch.nn.Linear(2, 4)
        model = accelerator.prepare(model)
        model_with_fp32_wrapper = accelerator.unwrap_model(model, keep_fp32_wrapper=True)
        input_tensor = torch.Tensor([1, 2]).to(dtype=torch.float16, device=accelerator.device)
        output = model_with_fp32_wrapper(input_tensor)
    if is_bf16_available():
        print('BF16 training check.')
        AcceleratorState._reset_state()
        dataloader_config = DataLoaderConfiguration(use_seedable_sampler=use_seedable_sampler)
        accelerator = Accelerator(mixed_precision='bf16', dataloader_config=dataloader_config)
        train_dl = generate_baseline_dataloader(train_set, generator, batch_size, use_seedable_sampler)
        model = RegressionModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        train_dl, model, optimizer = accelerator.prepare(train_dl, model, optimizer)
        set_seed(42)
        generator.manual_seed(42)
        for _ in range(3):
            for batch in train_dl:
                model.zero_grad()
                output = model(batch['x'])
                loss = torch.nn.functional.mse_loss(output, batch['y'])
                accelerator.backward(loss)
                optimizer.step()
        model = accelerator.unwrap_model(model).cpu()
        assert torch.allclose(old_model.a, model.a), 'Did not obtain the same model on CPU or distributed training.'
        assert torch.allclose(old_model.b, model.b), 'Did not obtain the same model on CPU or distributed training.'
    if is_ipex_available():
        print('ipex BF16 training check.')
        AcceleratorState._reset_state()
        dataloader_config = DataLoaderConfiguration(use_seedable_sampler=use_seedable_sampler)
        accelerator = Accelerator(mixed_precision='bf16', cpu=True, dataloader_config=dataloader_config)
        train_dl = generate_baseline_dataloader(train_set, generator, batch_size, use_seedable_sampler)
        model = RegressionModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        train_dl, model, optimizer = accelerator.prepare(train_dl, model, optimizer)
        set_seed(42)
        generator.manual_seed(42)
        for _ in range(3):
            for batch in train_dl:
                model.zero_grad()
                output = model(batch['x'])
                loss = torch.nn.functional.mse_loss(output, batch['y'])
                accelerator.backward(loss)
                optimizer.step()
        model = accelerator.unwrap_model(model).cpu()
        assert torch.allclose(old_model.a, model.a), 'Did not obtain the same model on CPU or distributed training.'
        assert torch.allclose(old_model.b, model.b), 'Did not obtain the same model on CPU or distributed training.'
    if is_xpu_available():
        print('xpu BF16 training check.')
        AcceleratorState._reset_state()
        dataloader_config = DataLoaderConfiguration(use_seedable_sampler=use_seedable_sampler)
        accelerator = Accelerator(mixed_precision='bf16', cpu=False, dataloader_config=dataloader_config)
        train_dl = generate_baseline_dataloader(train_set, generator, batch_size, use_seedable_sampler)
        model = RegressionModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        train_dl, model, optimizer = accelerator.prepare(train_dl, model, optimizer)
        set_seed(42)
        generator.manual_seed(42)
        for _ in range(3):
            for batch in train_dl:
                model.zero_grad()
                output = model(batch['x'])
                loss = torch.nn.functional.mse_loss(output, batch['y'])
                accelerator.backward(loss)
                optimizer.step()
        model = accelerator.unwrap_model(model).cpu()
        assert torch.allclose(old_model.a, model.a), 'Did not obtain the same model on XPU or distributed training.'
        assert torch.allclose(old_model.b, model.b), 'Did not obtain the same model on XPU or distributed training.'