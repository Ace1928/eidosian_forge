import os
from typing import Dict, Optional, Union
from ...configuration_utils import PretrainedConfig
from ...utils import add_start_docstrings, logging
from ..auto import CONFIG_MAPPING
@add_start_docstrings(BARK_SUBMODELCONFIG_START_DOCSTRING.format(config='BarkSemanticConfig', model='BarkSemanticModel'), '\n    Example:\n\n    ```python\n    >>> from transformers import BarkSemanticConfig, BarkSemanticModel\n\n    >>> # Initializing a Bark sub-module style configuration\n    >>> configuration = BarkSemanticConfig()\n\n    >>> # Initializing a model (with random weights) from the suno/bark style configuration\n    >>> model = BarkSemanticModel(configuration)\n\n    >>> # Accessing the model configuration\n    >>> configuration = model.config\n    ```')
class BarkSemanticConfig(BarkSubModelConfig):
    model_type = 'semantic'