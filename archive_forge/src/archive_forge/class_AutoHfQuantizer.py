import warnings
from typing import Dict, Optional, Union
from ..models.auto.configuration_auto import AutoConfig
from ..utils.quantization_config import (
from .quantizer_aqlm import AqlmHfQuantizer
from .quantizer_awq import AwqQuantizer
from .quantizer_bnb_4bit import Bnb4BitHfQuantizer
from .quantizer_bnb_8bit import Bnb8BitHfQuantizer
from .quantizer_gptq import GptqHfQuantizer
class AutoHfQuantizer:
    """
     The Auto-HF quantizer class that takes care of automatically instantiating to the correct
    `HfQuantizer` given the `QuantizationConfig`.
    """

    @classmethod
    def from_config(cls, quantization_config: Union[QuantizationConfigMixin, Dict], **kwargs):
        if isinstance(quantization_config, dict):
            quantization_config = AutoQuantizationConfig.from_dict(quantization_config)
        quant_method = quantization_config.quant_method
        if quant_method == QuantizationMethod.BITS_AND_BYTES:
            if quantization_config.load_in_8bit:
                quant_method += '_8bit'
            else:
                quant_method += '_4bit'
        if quant_method not in AUTO_QUANTIZER_MAPPING.keys():
            raise ValueError(f'Unknown quantization type, got {quant_method} - supported types are: {list(AUTO_QUANTIZER_MAPPING.keys())}')
        target_cls = AUTO_QUANTIZER_MAPPING[quant_method]
        return target_cls(quantization_config, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        quantization_config = AutoQuantizationConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        return cls.from_config(quantization_config)

    @classmethod
    def merge_quantization_configs(cls, quantization_config: Union[dict, QuantizationConfigMixin], quantization_config_from_args: Optional[QuantizationConfigMixin]):
        """
        handles situations where both quantization_config from args and quantization_config from model config are present.
        """
        if quantization_config_from_args is not None:
            warning_msg = "You passed `quantization_config` or equivalent parameters to `from_pretrained` but the model you're loading already has a `quantization_config` attribute. The `quantization_config` from the model will be used."
        else:
            warning_msg = ''
        if isinstance(quantization_config, dict):
            quantization_config = AutoQuantizationConfig.from_dict(quantization_config)
        if isinstance(quantization_config, (GPTQConfig, AwqConfig)) and quantization_config_from_args is not None:
            loading_attr_dict = quantization_config_from_args.get_loading_attributes()
            for attr, val in loading_attr_dict.items():
                setattr(quantization_config, attr, val)
            warning_msg += f'However, loading attributes (e.g. {list(loading_attr_dict.keys())}) will be overwritten with the one you passed to `from_pretrained`. The rest will be ignored.'
        if warning_msg != '':
            warnings.warn(warning_msg)
        return quantization_config