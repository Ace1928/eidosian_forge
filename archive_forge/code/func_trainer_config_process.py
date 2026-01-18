import copy
import importlib.metadata as importlib_metadata
import importlib.util
import weakref
from functools import partialmethod
from ..dependency_versions_check import dep_version_check
from ..utils import is_accelerate_available, is_torch_available, logging
def trainer_config_process(self, args, auto_find_batch_size=False):
    """
        Adjust the config with `TrainingArguments` values. This stage is run during `TrainingArguments` object
        creation.
        """
    train_batch_size = args.world_size * args.per_device_train_batch_size * args.gradient_accumulation_steps
    self.fill_match('train_micro_batch_size_per_gpu', args.per_device_train_batch_size, 'per_device_train_batch_size', not auto_find_batch_size)
    self.fill_match('gradient_accumulation_steps', args.gradient_accumulation_steps, 'gradient_accumulation_steps')
    self.fill_match('train_batch_size', train_batch_size, 'train_batch_size (calculated)', not auto_find_batch_size)
    self.fill_match('gradient_clipping', args.max_grad_norm, 'max_grad_norm')
    self.fill_match('optimizer.params.lr', args.learning_rate, 'learning_rate')
    self.fill_match('optimizer.params.betas', [args.adam_beta1, args.adam_beta2], 'adam_beta1+adam_beta2')
    self.fill_match('optimizer.params.eps', args.adam_epsilon, 'adam_epsilon')
    self.fill_match('optimizer.params.weight_decay', args.weight_decay, 'weight_decay')
    self.fill_only('scheduler.params.warmup_min_lr', 0)
    self.fill_match('scheduler.params.warmup_max_lr', args.learning_rate, 'learning_rate')
    if args.fp16 or args.fp16_full_eval:
        fp16_backend = 'apex' if args.fp16_backend == 'apex' else 'amp'
    else:
        fp16_backend = None
    if args.save_on_each_node:
        self.config['checkpoint'] = self.config.get('checkpoint', {})
        self.config['checkpoint']['use_node_local_storage'] = args.save_on_each_node
    self.fill_match('fp16.enabled', (args.fp16 or args.fp16_full_eval) and fp16_backend == 'amp', 'fp16|fp16_full_eval+fp16_backend(amp)')
    self.fill_match('amp.enabled', fp16_backend == 'apex', 'fp16+fp16_backend(apex)')
    self.fill_match('amp.opt_level', args.fp16_opt_level, 'fp16_opt_level')
    self.fill_match('bf16.enabled', args.bf16 or args.bf16_full_eval, 'bf16|bf16_full_eval')
    if self.is_true('bf16.enabled'):
        self._dtype = torch.bfloat16
    elif self.is_false('fp16.enabled'):
        self._dtype = torch.float32
    else:
        self._dtype = torch.float16