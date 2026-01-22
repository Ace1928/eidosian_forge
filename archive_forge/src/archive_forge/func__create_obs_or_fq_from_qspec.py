import copy
import torch
import warnings
from torch.fx import (
from torch.fx.graph import (
from torch.fx.node import Argument
from ..quantize import (
from ..observer import (
from ..qconfig import (
from ..qconfig_mapping import (
from .qconfig_mapping_utils import (
from .quantize_handler import (
from torch.ao.quantization import (
from torch.ao.quantization.utils import (
from ._equalize import (
from .pattern_utils import (
from .match_utils import (
from .utils import (
from torch.ao.quantization import (
from torch.ao.quantization.quantize import (
from ..utils import (
from ..backend_config.utils import (
from ..backend_config import (
from .custom_config import (
from torch.ao.quantization.quantizer import (
from torch.ao.quantization import ObserverOrFakeQuantize
from torch._subclasses import FakeTensor
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union
from dataclasses import asdict
def _create_obs_or_fq_from_qspec(quantization_spec: Optional[QuantizationSpecBase], obs_or_fq_map: Dict[EdgeOrNode, ObserverOrFakeQuantize], is_qat: bool):
    """ Create observer or fake quantize objects based on quantization spec

    Args:
       quantization_spec: used to store parameters to create the observer or fake quantizer
       obs_or_fq_map: this is a map from edge/output to the corresponding observer/fake_quant
       instance, it may be reused for different edge/output depending on configuration
    """
    if quantization_spec is None:
        return None
    if isinstance(quantization_spec, SharedQuantizationSpec):
        edge_or_node = quantization_spec.edge_or_node
        assert edge_or_node in obs_or_fq_map, f"please make sure only refer to edge or node that has observer/fake_quant inserted: '{edge_or_node}' not in\n{obs_or_fq_map.keys()}"
        return obs_or_fq_map[edge_or_node]
    elif isinstance(quantization_spec, DerivedQuantizationSpec):
        kwargs = {'dtype': quantization_spec.dtype, 'derive_qparams_fn': quantization_spec.derive_qparams_fn, 'quant_min': quantization_spec.quant_min, 'quant_max': quantization_spec.quant_max, 'qscheme': quantization_spec.qscheme, 'ch_axis': quantization_spec.ch_axis}
        edge_or_nodes = quantization_spec.derived_from
        obs_or_fqs = [obs_or_fq_map[k] for k in edge_or_nodes]
        kwargs['obs_or_fqs'] = obs_or_fqs
        return _DerivedObserverOrFakeQuantize.with_args(**kwargs)()
    elif isinstance(quantization_spec, FixedQParamsQuantizationSpec):
        kwargs = _get_observer_kwargs(quantization_spec)
        observer_ctr = FixedQParamsObserver.with_args(**kwargs)
        if is_qat:
            return FixedQParamsFakeQuantize.with_args(observer=observer_ctr)
        else:
            return observer_ctr()
    assert isinstance(quantization_spec, QuantizationSpec)
    observer_or_fake_quant_ctr = quantization_spec.observer_or_fake_quant_ctr
    kwargs = _get_observer_kwargs(quantization_spec)
    kwargs.pop('observer_or_fake_quant_ctr')
    obs_or_fq_class = observer_or_fake_quant_ctr
    if isinstance(observer_or_fake_quant_ctr, _PartialWrapper):
        obs_or_fq_class = observer_or_fake_quant_ctr.p.func
    if 'PerChannel' not in obs_or_fq_class.__name__:
        kwargs.pop('ch_axis')
    return observer_or_fake_quant_ctr.with_args(**kwargs)()