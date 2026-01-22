import functools
import inspect
import os
from collections import OrderedDict
from contextlib import contextmanager
from functools import partial
from typing import Any, Callable, Dict, Generator, Iterable, Optional, Sized, Tuple, Type, Union
from lightning_utilities.core.inheritance import get_all_subclasses
from torch.utils.data import BatchSampler, DataLoader, IterableDataset, Sampler
from typing_extensions import TypeGuard
from lightning_fabric.utilities.enums import LightningEnum
from lightning_fabric.utilities.exceptions import MisconfigurationException
from lightning_fabric.utilities.rank_zero import rank_zero_warn
from lightning_fabric.utilities.seed import pl_worker_init_function
def _dataloader_init_kwargs_resolve_sampler(dataloader: DataLoader, sampler: Union[Sampler, Iterable]) -> Dict[str, Any]:
    """This function is used to handle the sampler, batch_sampler arguments associated within a DataLoader for its re-
    instantiation."""
    batch_sampler = getattr(dataloader, 'batch_sampler')
    if batch_sampler is not None and type(batch_sampler) is not BatchSampler:
        batch_sampler_cls = type(batch_sampler)
        if hasattr(batch_sampler, '__pl_saved_args'):
            args = batch_sampler.__pl_saved_args
            kwargs = batch_sampler.__pl_saved_kwargs
            default_kwargs = batch_sampler.__pl_saved_default_kwargs
            arg_names = batch_sampler.__pl_saved_arg_names
            success, args, kwargs = _replace_value_in_saved_args('sampler', sampler, args, kwargs, default_kwargs, arg_names)
            if not success:
                raise TypeError(f'Trying to inject a modified sampler into the batch sampler; however, it seems the class `{batch_sampler_cls.__qualname__}` does not have an argument called `sampler.` To mitigate this, expose an argument `sampler` in the `__init__` method of your custom class.')
            batch_sampler = _reinstantiate_wrapped_cls(batch_sampler, *args, **kwargs)
        elif hasattr(batch_sampler, 'batch_size') and hasattr(batch_sampler, 'drop_last'):
            try:
                batch_sampler = batch_sampler_cls(sampler, batch_size=batch_sampler.batch_size, drop_last=batch_sampler.drop_last)
            except TypeError as ex:
                import re
                match = re.match('.*__init__\\(\\) (got multiple values)|(missing \\d required)', str(ex))
                if not match:
                    raise
                raise TypeError(" Lightning can't inject a (distributed) sampler into your batch sampler, because it doesn't subclass PyTorch's `BatchSampler`. To mitigate this, either follow the API of `BatchSampler` or set`.setup_dataloaders(..., use_distributed_sampler=False)`. If you choose the latter, you will be responsible for handling the distributed sampling within your batch sampler.") from ex
        else:
            raise TypeError(" Lightning can't inject a (distributed) sampler into your batch sampler, because it doesn't subclass PyTorch's `BatchSampler`. To mitigate this, either follow the API of `BatchSampler` or set`.setup_dataloaders(..., use_distributed_sampler=False)`. If you choose the latter, you will be responsible for handling the distributed sampling within your batch sampler.")
        return {'sampler': None, 'shuffle': False, 'batch_sampler': batch_sampler, 'batch_size': 1, 'drop_last': False}
    return {'sampler': sampler, 'shuffle': False, 'batch_sampler': None}