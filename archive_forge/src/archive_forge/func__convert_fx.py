from typing import Any, Dict, Optional, Tuple, Union
import warnings
import torch
import copy
from torch.fx import GraphModule
from torch.fx.graph_module import _USER_PRESERVED_ATTRIBUTES_KEY
from .fx.tracer import QuantizationTracer
from .fx.tracer import (  # noqa: F401
from .fx.fuse import fuse  # noqa: F401
from .fx.prepare import prepare  # noqa: F401
from .fx.convert import convert
from .backend_config import (  # noqa: F401
from .fx.graph_module import ObservedGraphModule  # noqa: F401
from .fx.custom_config import (
from .fx.utils import get_custom_module_class_keys  # noqa: F401
from .fx.utils import get_skipped_module_name_and_classes
from .qconfig_mapping import QConfigMapping
def _convert_fx(graph_module: GraphModule, is_reference: bool, convert_custom_config: Union[ConvertCustomConfig, Dict[str, Any], None]=None, is_standalone_module: bool=False, _remove_qconfig: bool=True, qconfig_mapping: Union[QConfigMapping, Dict[str, Any], None]=None, backend_config: Union[BackendConfig, Dict[str, Any], None]=None, is_decomposed: bool=False) -> GraphModule:
    """ `is_standalone_module`: see docs in :func:`~torch.ao.quantization.prepare_standalone_module_fx`
    """
    if convert_custom_config is None:
        convert_custom_config = ConvertCustomConfig()
    if isinstance(convert_custom_config, Dict):
        warnings.warn('Passing a convert_custom_config_dict to convert is deprecated and will not be supported in a future version. Please pass in a ConvertCustomConfig instead.')
        convert_custom_config = ConvertCustomConfig.from_dict(convert_custom_config)
    _check_is_graph_module(graph_module)
    preserved_attr_names = convert_custom_config.preserved_attributes
    preserved_attrs = {attr: getattr(graph_module, attr) for attr in preserved_attr_names if hasattr(graph_module, attr)}
    quantized = convert(graph_module, is_reference, convert_custom_config, is_standalone_module, _remove_qconfig_flag=_remove_qconfig, qconfig_mapping=qconfig_mapping, backend_config=backend_config, is_decomposed=is_decomposed)
    attach_preserved_attrs_to_model(quantized, preserved_attrs)
    return quantized