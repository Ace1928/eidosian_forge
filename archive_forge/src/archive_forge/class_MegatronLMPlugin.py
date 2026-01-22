import argparse
import copy
import enum
import functools
import os
import typing
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Tuple, get_args
import torch
from .constants import FSDP_AUTO_WRAP_POLICY, FSDP_BACKWARD_PREFETCH, FSDP_SHARDING_STRATEGY, FSDP_STATE_DICT_TYPE
from .environment import str_to_bool
from .imports import is_cuda_available, is_npu_available, is_xpu_available
from .versions import compare_versions
@dataclass
class MegatronLMPlugin:
    """
    Plugin for Megatron-LM to enable tensor, pipeline, sequence and data parallelism. Also to enable selective
    activation recomputation and optimized fused kernels.
    """
    tp_degree: int = field(default=None, metadata={'help': 'tensor parallelism degree.'})
    pp_degree: int = field(default=None, metadata={'help': 'pipeline parallelism degree.'})
    num_micro_batches: int = field(default=None, metadata={'help': 'number of micro-batches.'})
    gradient_clipping: float = field(default=None, metadata={'help': 'gradient clipping value based on global L2 Norm (0 to disable)'})
    sequence_parallelism: bool = field(default=None, metadata={'help': 'enable sequence parallelism'})
    recompute_activations: bool = field(default=None, metadata={'help': 'enable selective activation recomputation'})
    use_distributed_optimizer: bool = field(default=None, metadata={'help': 'enable distributed optimizer'})
    pipeline_model_parallel_split_rank: int = field(default=None, metadata={'help': 'Rank where encoder and decoder should be split.'})
    num_layers_per_virtual_pipeline_stage: int = field(default=None, metadata={'help': 'Number of layers per virtual pipeline stage.'})
    is_train_batch_min: str = field(default=True, metadata={'help': 'If both train & eval dataloaders are specified, this will decide the micro_batch_size'})
    train_iters: int = field(default=None, metadata={'help': 'Total number of iterations to train over all training runs. Note that either train-iters or train-samples should be provided when using `MegatronLMDummyScheduler`'})
    train_samples: int = field(default=None, metadata={'help': 'Total number of samples to train over all training runs. Note that either train-iters or train-samples should be provided when using `MegatronLMDummyScheduler`'})
    weight_decay_incr_style: str = field(default='constant', metadata={'help': 'Weight decay increment function. choices=["constant", "linear", "cosine"]. '})
    start_weight_decay: float = field(default=None, metadata={'help': 'Initial weight decay coefficient for L2 regularization.'})
    end_weight_decay: float = field(default=None, metadata={'help': 'End of run weight decay coefficient for L2 regularization.'})
    lr_decay_style: str = field(default='linear', metadata={'help': "Learning rate decay function. choices=['constant', 'linear', 'cosine']."})
    lr_decay_iters: int = field(default=None, metadata={'help': 'Number of iterations for learning rate decay. If None defaults to `train_iters`.'})
    lr_decay_samples: int = field(default=None, metadata={'help': 'Number of samples for learning rate decay. If None defaults to `train_samples`.'})
    lr_warmup_iters: int = field(default=None, metadata={'help': 'number of iterations to linearly warmup learning rate over.'})
    lr_warmup_samples: int = field(default=None, metadata={'help': 'number of samples to linearly warmup learning rate over.'})
    lr_warmup_fraction: float = field(default=None, metadata={'help': 'fraction of lr-warmup-(iters/samples) to linearly warmup learning rate over.'})
    min_lr: float = field(default=0, metadata={'help': 'Minumum value for learning rate. The scheduler clip values below this threshold.'})
    consumed_samples: List[int] = field(default=None, metadata={'help': 'Number of samples consumed in the same order as the dataloaders to `accelerator.prepare` call.'})
    no_wd_decay_cond: Optional[Callable] = field(default=None, metadata={'help': 'Condition to disable weight decay.'})
    scale_lr_cond: Optional[Callable] = field(default=None, metadata={'help': 'Condition to scale learning rate.'})
    lr_mult: float = field(default=1.0, metadata={'help': 'Learning rate multiplier.'})
    megatron_dataset_flag: bool = field(default=False, metadata={'help': 'Whether the format of dataset follows Megatron-LM Indexed/Cached/MemoryMapped format.'})
    seq_length: int = field(default=None, metadata={'help': 'Maximum sequence length to process.'})
    encoder_seq_length: int = field(default=None, metadata={'help': 'Maximum sequence length to process for the encoder.'})
    decoder_seq_length: int = field(default=None, metadata={'help': 'Maximum sequence length to process for the decoder.'})
    tensorboard_dir: str = field(default=None, metadata={'help': 'Path to save tensorboard logs.'})
    set_all_logging_options: bool = field(default=False, metadata={'help': 'Whether to set all logging options.'})
    eval_iters: int = field(default=100, metadata={'help': 'Number of iterations to run for evaluation validation/test for.'})
    eval_interval: int = field(default=1000, metadata={'help': 'Interval between running evaluation on validation set.'})
    return_logits: bool = field(default=False, metadata={'help': 'Whether to return logits from the model.'})
    custom_train_step_class: Optional[Any] = field(default=None, metadata={'help': 'Custom train step class.'})
    custom_train_step_kwargs: Optional[Dict[str, Any]] = field(default=None, metadata={'help': 'Custom train step kwargs.'})
    custom_model_provider_function: Optional[Callable] = field(default=None, metadata={'help': 'Custom model provider function.'})
    custom_prepare_model_function: Optional[Callable] = field(default=None, metadata={'help': 'Custom prepare model function.'})
    other_megatron_args: Optional[Dict[str, Any]] = field(default=None, metadata={'help': 'Other Megatron-LM arguments. Please refer Megatron-LM'})

    def __post_init__(self):
        prefix = 'MEGATRON_LM_'
        if self.tp_degree is None:
            self.tp_degree = int(os.environ.get(prefix + 'TP_DEGREE', 1))
        if self.pp_degree is None:
            self.pp_degree = int(os.environ.get(prefix + 'PP_DEGREE', 1))
        if self.num_micro_batches is None:
            self.num_micro_batches = int(os.environ.get(prefix + 'NUM_MICRO_BATCHES', 1))
        if self.gradient_clipping is None:
            self.gradient_clipping = float(os.environ.get(prefix + 'GRADIENT_CLIPPING', 1.0))
        if self.recompute_activations is None:
            self.recompute_activations = str_to_bool(os.environ.get(prefix + 'RECOMPUTE_ACTIVATIONS', 'False')) == 1
        if self.use_distributed_optimizer is None:
            self.use_distributed_optimizer = str_to_bool(os.environ.get(prefix + 'USE_DISTRIBUTED_OPTIMIZER', 'False')) == 1
        if self.sequence_parallelism is None:
            self.sequence_parallelism = str_to_bool(os.environ.get(prefix + 'SEQUENCE_PARALLELISM', 'False')) == 1
        if self.pp_degree > 1 or self.use_distributed_optimizer:
            self.DDP_impl = 'local'
        else:
            self.DDP_impl = 'torch'
        if self.consumed_samples is not None:
            if len(self.consumed_samples) == 1:
                self.consumed_samples.extend([0, 0])
            elif len(self.consumed_samples) == 2:
                self.consumed_samples.append(0)
        self.megatron_lm_default_args = {'tensor_model_parallel_size': self.tp_degree, 'pipeline_model_parallel_size': self.pp_degree, 'pipeline_model_parallel_split_rank': self.pipeline_model_parallel_split_rank, 'num_layers_per_virtual_pipeline_stage': self.num_layers_per_virtual_pipeline_stage, 'DDP_impl': self.DDP_impl, 'use_distributed_optimizer': self.use_distributed_optimizer, 'sequence_parallel': self.sequence_parallelism, 'clip_grad': self.gradient_clipping, 'num_micro_batches': self.num_micro_batches, 'consumed_samples': self.consumed_samples, 'no_wd_decay_cond': self.no_wd_decay_cond, 'scale_lr_cond': self.scale_lr_cond, 'lr_mult': self.lr_mult, 'megatron_dataset_flag': self.megatron_dataset_flag, 'eval_iters': self.eval_iters, 'eval_interval': self.eval_interval}
        if self.recompute_activations:
            self.megatron_lm_default_args['recompute_granularity'] = 'selective'
        if self.tensorboard_dir is not None:
            self.megatron_lm_default_args['tensorboard_dir'] = self.tensorboard_dir
            if self.set_all_logging_options:
                self.set_tensorboard_logging_options()
        if self.other_megatron_args is not None:
            self.megatron_lm_default_args.update(self.other_megatron_args)

    def set_network_size_args(self, model, batch_data=None):
        if 'megatron-bert' in model.config.model_type.lower():
            model_type_name = 'bert'
            num_layers = model.config.num_hidden_layers
            hidden_size = model.config.hidden_size
            num_attention_heads = model.config.num_attention_heads
            max_position_embeddings = model.config.max_position_embeddings
            num_labels = model.config.num_labels
            orig_vocab_size = model.config.vocab_size
            if 'maskedlm' in model.__class__.__name__.lower():
                pretraining_flag = True
            if self.seq_length is not None:
                if self.encoder_seq_length is not None:
                    warnings.warn('Both `seq_length` and `encoder_seq_length` are set. Using `encoder_seq_length`.')
                self.seq_length = self.encoder_seq_length
            elif self.encoder_seq_length is not None:
                self.seq_length = self.encoder_seq_length
            elif batch_data is not None:
                self.seq_length = batch_data['input_ids'].shape[1]
            else:
                self.seq_length = max_position_embeddings
            self.megatron_lm_default_args['seq_length'] = self.seq_length
        elif 'gpt2' in model.config.model_type.lower():
            model_type_name = 'gpt'
            num_layers = model.config.n_layer
            hidden_size = model.config.n_embd
            num_attention_heads = model.config.n_head
            max_position_embeddings = model.config.n_positions
            orig_vocab_size = model.config.vocab_size
            pretraining_flag = True
            if self.seq_length is not None:
                if self.decoder_seq_length is not None:
                    warnings.warn('Both `seq_length` and `decoder_seq_length` are set. Using `decoder_seq_length`.')
                self.seq_length = self.decoder_seq_length
            elif self.decoder_seq_length is not None:
                self.seq_length = self.decoder_seq_length
            elif batch_data is not None:
                self.seq_length = batch_data['input_ids'].shape[1]
            else:
                self.seq_length = max_position_embeddings
            self.megatron_lm_default_args['seq_length'] = self.seq_length
            self.megatron_lm_default_args['return_logits'] = self.return_logits
            self.megatron_lm_default_args['tokenizer_type'] = 'GPT2BPETokenizer'
        elif 't5' in model.config.model_type.lower():
            model_type_name = 't5'
            num_layers = model.config.num_layers
            hidden_size = model.config.d_model
            num_attention_heads = model.config.num_heads
            max_position_embeddings = model.config.n_positions if hasattr(model.config, 'n_positions') else 1024
            orig_vocab_size = model.config.vocab_size
            pretraining_flag = True
            if self.encoder_seq_length is None:
                if batch_data is not None:
                    self.encoder_seq_length = batch_data['input_ids'].shape[1]
                else:
                    self.encoder_seq_length = max_position_embeddings
            if self.decoder_seq_length is None:
                if batch_data is not None:
                    self.decoder_seq_length = batch_data['labels'].shape[1]
                else:
                    self.decoder_seq_length = max_position_embeddings
            self.megatron_lm_default_args['encoder_seq_length'] = self.encoder_seq_length
            self.megatron_lm_default_args['decoder_seq_length'] = self.decoder_seq_length
        else:
            raise ValueError('🤗 Accelerate Megatron-LM integration supports only BERT, GPT and T5 model. Please check the model you are using is one of those.')
        self.megatron_lm_default_args['model_type_name'] = model_type_name
        self.megatron_lm_default_args['num_layers'] = num_layers
        self.megatron_lm_default_args['hidden_size'] = hidden_size
        self.megatron_lm_default_args['num_attention_heads'] = num_attention_heads
        self.megatron_lm_default_args['max_position_embeddings'] = max_position_embeddings
        self.megatron_lm_default_args['pretraining_flag'] = pretraining_flag
        self.megatron_lm_default_args['orig_vocab_size'] = orig_vocab_size
        self.megatron_lm_default_args['model_return_dict'] = model.config.return_dict
        if model_type_name == 'bert':
            self.megatron_lm_default_args['num_labels'] = num_labels

    def set_mixed_precision(self, mixed_precision):
        if mixed_precision == 'fp16':
            self.megatron_lm_default_args['fp16'] = True
        elif mixed_precision == 'bf16':
            self.megatron_lm_default_args['bf16'] = True
            self.DDP_impl = 'local'
            self.megatron_lm_default_args['DDP_impl'] = self.DDP_impl

    def set_training_args(self, micro_batch_size, dp_degree):
        self.data_parallel_size = dp_degree
        self.micro_batch_size = micro_batch_size
        self.global_batch_size = dp_degree * micro_batch_size * self.num_micro_batches
        self.megatron_lm_default_args['data_parallel_size'] = self.data_parallel_size
        self.megatron_lm_default_args['micro_batch_size'] = self.micro_batch_size
        self.megatron_lm_default_args['global_batch_size'] = self.global_batch_size

    def set_optimizer_type(self, optimizer):
        optimizer_name = optimizer.__class__.__name__.lower()
        if 'adam' in optimizer_name:
            self.megatron_lm_default_args['optimizer'] = 'adam'
            self.megatron_lm_default_args['adam_beta1'] = optimizer.defaults['betas'][0]
            self.megatron_lm_default_args['adam_beta2'] = optimizer.defaults['betas'][1]
            self.megatron_lm_default_args['adam_eps'] = optimizer.defaults['eps']
        elif 'sgd' in optimizer_name:
            self.megatron_lm_default_args['optimizer'] = 'sgd'
            self.megatron_lm_default_args['sgd_momentum'] = optimizer.defaults['momentum']
        else:
            raise ValueError(f'Optimizer {optimizer_name} is not supported by Megatron-LM')
        self.megatron_lm_default_args['lr'] = optimizer.defaults['lr']
        self.megatron_lm_default_args['weight_decay'] = optimizer.defaults['weight_decay']

    def set_scheduler_args(self, scheduler):
        if self.train_iters is None:
            self.train_iters = scheduler.total_num_steps // self.megatron_lm_default_args['data_parallel_size']
            if self.train_samples is not None:
                self.train_samples = None
                warnings.warn('Ignoring `train_samples` as `train_iters` based on scheduler is being used for training.')
        if self.lr_warmup_iters is None:
            self.lr_warmup_iters = scheduler.warmup_num_steps // self.megatron_lm_default_args['data_parallel_size']
            if self.lr_warmup_samples is not None:
                warnings.warn('Ignoring `lr_warmup_samples` as `lr_warmup_iters` based on scheduler is being used for training.')
            self.lr_warmup_samples = 0
        self.megatron_lm_default_args['train_iters'] = self.train_iters
        self.megatron_lm_default_args['lr_warmup_iters'] = self.lr_warmup_iters
        self.megatron_lm_default_args['train_samples'] = self.train_samples
        self.megatron_lm_default_args['lr_warmup_samples'] = self.lr_warmup_samples
        self.megatron_lm_default_args['lr_decay_iters'] = self.lr_decay_iters
        self.megatron_lm_default_args['lr_decay_samples'] = self.lr_decay_samples
        self.megatron_lm_default_args['lr_warmup_fraction'] = self.lr_warmup_fraction
        self.megatron_lm_default_args['lr_decay_style'] = self.lr_decay_style
        self.megatron_lm_default_args['weight_decay_incr_style'] = self.weight_decay_incr_style
        self.megatron_lm_default_args['start_weight_decay'] = self.start_weight_decay
        self.megatron_lm_default_args['end_weight_decay'] = self.end_weight_decay
        self.megatron_lm_default_args['min_lr'] = self.min_lr

    def set_tensorboard_logging_options(self):
        from megatron.arguments import _add_logging_args
        parser = argparse.ArgumentParser()
        parser = _add_logging_args(parser)
        logging_args = parser.parse_known_args()
        self.dataset_args = vars(logging_args[0])
        for key, value in self.dataset_args.items():
            if key.startswith('log_'):
                self.megatron_lm_default_args[key] = True
            elif key.startswith('no_log_'):
                self.megatron_lm_default_args[key.replace('no_', '')] = True