import random
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
from packaging import version
from transformers.utils import is_tf_available
from ...utils import (
from ...utils.normalized_config import NormalizedConfigManager
from .base import ConfigBehavior, OnnxConfig, OnnxConfigWithPast, OnnxSeq2SeqConfigWithPast
from .config import (
from .model_patcher import (
class ASTOnnxConfig(OnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedConfig.with_args(num_mel_bins='num_mel_bins', max_length='max_length', allow_new=True)
    DUMMY_INPUT_GENERATOR_CLASSES = (ASTDummyAudioInputGenerator,)
    ATOL_FOR_VALIDATION = 0.0001

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        return {'input_values': {0: 'batch_size'}}