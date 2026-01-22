import os
from typing import Union
from ...configuration_utils import PretrainedConfig
from ...models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from ...utils import logging
from ..auto import CONFIG_MAPPING

        Instantiate a [`InstructBlipConfig`] (or a derived class) from a InstructBLIP vision model, Q-Former and
        language model configurations.

        Returns:
            [`InstructBlipConfig`]: An instance of a configuration object
        