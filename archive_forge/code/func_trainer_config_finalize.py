import copy
import importlib.metadata as importlib_metadata
import importlib.util
import weakref
from functools import partialmethod
from ..dependency_versions_check import dep_version_check
from ..utils import is_accelerate_available, is_torch_available, logging
def trainer_config_finalize(self, args, model, num_training_steps):
    """
        This stage is run after we have the model and know num_training_steps.

        Now we can complete the configuration process.
        """
    hidden_size_based_keys = ['zero_optimization.reduce_bucket_size', 'zero_optimization.stage3_prefetch_bucket_size', 'zero_optimization.stage3_param_persistence_threshold']
    hidden_size_auto_keys = [x for x in hidden_size_based_keys if self.is_auto(x)]
    if len(hidden_size_auto_keys) > 0:
        if hasattr(model.config, 'hidden_size'):
            hidden_size = model.config.hidden_size
        elif hasattr(model.config, 'hidden_sizes'):
            hidden_size = max(model.config.hidden_sizes)
        else:
            raise ValueError(f"The model's config file has neither `hidden_size` nor `hidden_sizes` entry, therefore it's not possible to automatically fill out the following `auto` entries in the DeepSpeed config file: {hidden_size_auto_keys}. You can fix that by replacing `auto` values for these keys with an integer value of your choice.")
        self.fill_only('zero_optimization.reduce_bucket_size', hidden_size * hidden_size)
        if self.is_zero3():
            self.fill_only('zero_optimization.stage3_prefetch_bucket_size', 0.9 * hidden_size * hidden_size)
            self.fill_only('zero_optimization.stage3_param_persistence_threshold', 10 * hidden_size)
    self.fill_match('scheduler.params.total_num_steps', num_training_steps, 'num_training_steps (calculated)')
    self.fill_match('scheduler.params.warmup_num_steps', args.get_warmup_steps(num_training_steps), 'warmup_steps')
    if len(self.mismatches) > 0:
        mismatches = '\n'.join(self.mismatches)
        raise ValueError(f"Please correct the following DeepSpeed config values that mismatch TrainingArguments values:\n{mismatches}\nThe easiest method is to set these DeepSpeed config values to 'auto'.")