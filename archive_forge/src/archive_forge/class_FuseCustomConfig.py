from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type
from torch.ao.quantization import QConfigMapping
from torch.ao.quantization.backend_config import BackendConfig
from torch.ao.quantization.quant_type import QuantType, _quant_type_from_str, _get_quant_type_to_str
class FuseCustomConfig:
    """
    Custom configuration for :func:`~torch.ao.quantization.quantize_fx.fuse_fx`.

    Example usage::

        fuse_custom_config = FuseCustomConfig().set_preserved_attributes(["attr1", "attr2"])
    """

    def __init__(self):
        self.preserved_attributes: List[str] = []

    def __repr__(self):
        dict_nonempty = {k: v for k, v in self.__dict__.items() if len(v) > 0}
        return f'FuseCustomConfig({dict_nonempty})'

    def set_preserved_attributes(self, attributes: List[str]) -> FuseCustomConfig:
        """
        Set the names of the attributes that will persist in the graph module even if they are not used in
        the model's ``forward`` method.
        """
        self.preserved_attributes = attributes
        return self

    @classmethod
    def from_dict(cls, fuse_custom_config_dict: Dict[str, Any]) -> FuseCustomConfig:
        """
        Create a ``ConvertCustomConfig`` from a dictionary with the following items:

            "preserved_attributes": a list of attributes that persist even if they are not used in ``forward``

        This function is primarily for backward compatibility and may be removed in the future.
        """
        conf = cls()
        conf.set_preserved_attributes(fuse_custom_config_dict.get(PRESERVED_ATTRIBUTES_DICT_KEY, []))
        return conf

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert this ``FuseCustomConfig`` to a dictionary with the items described in
        :func:`~torch.ao.quantization.fx.custom_config.ConvertCustomConfig.from_dict`.
        """
        d: Dict[str, Any] = {}
        if len(self.preserved_attributes) > 0:
            d[PRESERVED_ATTRIBUTES_DICT_KEY] = self.preserved_attributes
        return d