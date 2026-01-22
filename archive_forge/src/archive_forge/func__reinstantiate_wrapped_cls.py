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
def _reinstantiate_wrapped_cls(orig_object: Any, *args: Any, explicit_cls: Optional[Type]=None, **kwargs: Any) -> Any:
    constructor = type(orig_object) if explicit_cls is None else explicit_cls
    try:
        result = constructor(*args, **kwargs)
    except TypeError as ex:
        import re
        match = re.match(".*__init__\\(\\) got multiple values .* '(\\w+)'", str(ex))
        if not match:
            raise
        argument = match.groups()[0]
        message = f"The {constructor.__name__} implementation has an error where more than one `__init__` argument can be passed to its parent's `{argument}=...` `__init__` argument. This is likely caused by allowing passing both a custom argument that will map to the `{argument}` argument as well as `**kwargs`. `kwargs` should be filtered to make sure they don't contain the `{argument}` key. This argument was automatically passed to your object by PyTorch Lightning."
        raise MisconfigurationException(message) from ex
    attrs_record = getattr(orig_object, '__pl_attrs_record', [])
    for args, fn in attrs_record:
        fn(result, *args)
    return result