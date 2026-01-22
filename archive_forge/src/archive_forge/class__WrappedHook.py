from collections import OrderedDict, namedtuple
import itertools
import warnings
import functools
import weakref
import torch
from torch._prims_common import DeviceLikeType
from ..parameter import Parameter
import torch.utils.hooks as hooks
from torch import Tensor, device, dtype
from typing import Union, Tuple, Any, Callable, Iterator, Set, Optional, overload, TypeVar, Mapping, Dict, List
from typing_extensions import Self
from ...utils.hooks import RemovableHandle
class _WrappedHook:

    def __init__(self, hook: Callable, module: Optional['Module']=None):
        self.hook: Callable = hook
        functools.update_wrapper(self, hook)
        self.with_module: bool = False
        if module is not None:
            self.module: weakref.ReferenceType[Module] = weakref.ref(module)
            self.with_module = True

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self.with_module:
            module = self.module()
            if module is None:
                raise RuntimeError('You are trying to call the hook of a dead Module!')
            return self.hook(module, *args, **kwargs)
        return self.hook(*args, **kwargs)

    def __getstate__(self) -> Dict:
        result = {'hook': self.hook, 'with_module': self.with_module}
        if self.with_module:
            result['module'] = self.module()
        return result

    def __setstate__(self, state: Dict):
        self.hook = state['hook']
        self.with_module = state['with_module']
        if self.with_module:
            if state['module'] is None:
                raise RuntimeError('You are trying to revive the hook of a dead Module!')
            self.module = weakref.ref(state['module'])