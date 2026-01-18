import argparse
import gc
import json
import os
import torch
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup, set_seed
from accelerate import Accelerator, DistributedType
from accelerate.utils import is_npu_available, is_xpu_available
from accelerate.utils.deepspeed import DummyOptim, DummyScheduler
def training_function(config, args):
    accelerator = Accelerator()
    lr = config['lr']
    num_epochs = int(config['num_epochs'])
    seed = int(config['seed'])
    batch_size = int(config['batch_size'])
    model_name = args.model_name_or_path
    set_seed(seed)
    train_dataloader, eval_dataloader = get_dataloaders(accelerator, batch_size, model_name, args.n_train, args.n_val)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, return_dict=True)
    optimizer_cls = AdamW if accelerator.state.deepspeed_plugin is None or 'optimizer' not in accelerator.state.deepspeed_plugin.deepspeed_config else DummyOptim
    optimizer = optimizer_cls(params=model.parameters(), lr=lr)
    if accelerator.state.deepspeed_plugin is not None:
        gradient_accumulation_steps = accelerator.state.deepspeed_plugin.deepspeed_config['gradient_accumulation_steps']
    else:
        gradient_accumulation_steps = 1
    max_training_steps = len(train_dataloader) * num_epochs // gradient_accumulation_steps
    if accelerator.state.deepspeed_plugin is None or 'scheduler' not in accelerator.state.deepspeed_plugin.deepspeed_config:
        lr_scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=0, num_training_steps=max_training_steps)
    else:
        lr_scheduler = DummyScheduler(optimizer, total_num_steps=max_training_steps, warmup_num_steps=0)
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(model, optimizer, train_dataloader, eval_dataloader, lr_scheduler)
    overall_step = 0
    starting_epoch = 0
    train_total_peak_memory = {}
    for epoch in range(starting_epoch, num_epochs):
        with TorchTracemalloc() as tracemalloc:
            model.train()
            for step, batch in enumerate(train_dataloader):
                outputs = model(**batch)
                loss = outputs.loss
                loss = loss / gradient_accumulation_steps
                accelerator.backward(loss)
                if step % gradient_accumulation_steps == 0:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                overall_step += 1
        accelerator.print(f'Memory before entering the train : {b2mb(tracemalloc.begin)}')
        accelerator.print(f'Memory consumed at the end of the train (end-begin): {tracemalloc.used}')
        accelerator.print(f'Peak Memory consumed during the train (max-begin): {tracemalloc.peaked}')
        accelerator.print(f'Total Peak Memory consumed during the train (max): {tracemalloc.peaked + b2mb(tracemalloc.begin)}')
        train_total_peak_memory[f'epoch-{epoch}'] = tracemalloc.peaked + b2mb(tracemalloc.begin)
        if args.peak_memory_upper_bound is not None:
            assert train_total_peak_memory[f'epoch-{epoch}'] <= args.peak_memory_upper_bound, 'Peak memory usage exceeded the upper bound'
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        with open(os.path.join(args.output_dir, 'peak_memory_utilization.json'), 'w') as f:
            json.dump(train_total_peak_memory, f)