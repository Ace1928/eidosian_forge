import os
from typing import Dict, Optional, Union
from ...configuration_utils import PretrainedConfig
from ...utils import add_start_docstrings, logging
from ..auto import CONFIG_MAPPING
@add_start_docstrings(BARK_SUBMODELCONFIG_START_DOCSTRING.format(config='BarkFineConfig', model='BarkFineModel'), '\n        n_codes_total (`int`, *optional*, defaults to 8):\n            The total number of audio codebooks predicted. Used in the fine acoustics sub-model.\n        n_codes_given (`int`, *optional*, defaults to 1):\n            The number of audio codebooks predicted in the coarse acoustics sub-model. Used in the acoustics\n            sub-models.\n    Example:\n\n    ```python\n    >>> from transformers import BarkFineConfig, BarkFineModel\n\n    >>> # Initializing a Bark sub-module style configuration\n    >>> configuration = BarkFineConfig()\n\n    >>> # Initializing a model (with random weights) from the suno/bark style configuration\n    >>> model = BarkFineModel(configuration)\n\n    >>> # Accessing the model configuration\n    >>> configuration = model.config\n    ```')
class BarkFineConfig(BarkSubModelConfig):
    model_type = 'fine_acoustics'

    def __init__(self, tie_word_embeddings=True, n_codes_total=8, n_codes_given=1, **kwargs):
        self.n_codes_total = n_codes_total
        self.n_codes_given = n_codes_given
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)