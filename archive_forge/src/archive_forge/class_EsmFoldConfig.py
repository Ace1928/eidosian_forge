from dataclasses import asdict, dataclass
from typing import Optional
from ...configuration_utils import PretrainedConfig
from ...utils import logging
@dataclass
class EsmFoldConfig:
    esm_type: str = None
    fp16_esm: bool = True
    use_esm_attn_map: bool = False
    esm_ablate_pairwise: bool = False
    esm_ablate_sequence: bool = False
    esm_input_dropout: float = 0
    embed_aa: bool = True
    bypass_lm: bool = False
    lddt_head_hid_dim: int = 128
    trunk: 'TrunkConfig' = None

    def __post_init__(self):
        if self.trunk is None:
            self.trunk = TrunkConfig()
        elif isinstance(self.trunk, dict):
            self.trunk = TrunkConfig(**self.trunk)

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = asdict(self)
        output['trunk'] = self.trunk.to_dict()
        return output