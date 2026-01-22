from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type
from torch.ao.quantization import QConfigMapping
from torch.ao.quantization.backend_config import BackendConfig
from torch.ao.quantization.quant_type import QuantType, _quant_type_from_str, _get_quant_type_to_str
def _get_qconfig_mapping(obj: Any, dict_key: str) -> Optional[QConfigMapping]:
    """
            Convert the given object into a QConfigMapping if possible, else throw an exception.
            """
    if isinstance(obj, QConfigMapping) or obj is None:
        return obj
    if isinstance(obj, Dict):
        return QConfigMapping.from_dict(obj)
    raise ValueError(f"""Expected QConfigMapping in prepare_custom_config_dict["{dict_key}"], got '{type(obj)}'""")