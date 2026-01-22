from .fake_quantize import *  # noqa: F403
from .fuse_modules import fuse_modules  # noqa: F403
from .fuse_modules import fuse_modules_qat  # noqa: F403
from .fuser_method_mappings import *  # noqa: F403
from .observer import *  # noqa: F403
from .qconfig import *  # noqa: F403
from .qconfig_mapping import *  # noqa: F403
from .quant_type import *  # noqa: F403
from .quantization_mappings import *  # type: ignore[no-redef]
from .quantize import *  # noqa: F403
from .quantize_jit import *  # noqa: F403
from .stubs import *  # noqa: F403
from .pt2e.eval_utils import _move_exported_model_to_eval as move_exported_model_to_eval
from .pt2e.generate_numeric_debug_handle import generate_numeric_debug_handle  # noqa: F401
from typing import Union, List, Callable, Tuple, Optional
from torch import Tensor
import torch
class _DerivedObserverOrFakeQuantize(ObserverBase):
    """This observer is used to describe an observer whose quantization parameters
    are derived from other observers
    """

    def __init__(self, dtype: torch.dtype, obs_or_fqs: List[ObserverOrFakeQuantize], derive_qparams_fn: Callable[[List[ObserverOrFakeQuantize]], Tuple[Tensor, Tensor]], quant_min: Optional[int]=None, quant_max: Optional[int]=None, qscheme: Optional[torch.qscheme]=None, ch_axis: Optional[int]=None):
        super().__init__(dtype)
        self.obs_or_fqs = obs_or_fqs
        self.derive_qparams_fn = derive_qparams_fn
        self.quant_min = quant_min
        self.quant_max = quant_max
        self.qscheme = qscheme
        self.ch_axis = ch_axis
        from .utils import is_per_channel
        if is_per_channel(self.qscheme):
            assert self.ch_axis is not None, 'Must provide a valid ch_axis if qscheme is per channel'

    def forward(self, x: Tensor) -> Tensor:
        return x

    def calculate_qparams(self):
        return self.derive_qparams_fn(self.obs_or_fqs)