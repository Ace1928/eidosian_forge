import copy
from typing import Dict
from ...generation.configuration_utils import GenerationConfig
from ...utils import logging
class BarkFineGenerationConfig(GenerationConfig):
    model_type = 'fine_acoustics'

    def __init__(self, temperature=1.0, max_fine_history_length=512, max_fine_input_length=1024, n_fine_codebooks=8, **kwargs):
        """Class that holds a generation configuration for [`BarkFineModel`].

        [`BarkFineModel`] is an autoencoder model, so should not usually be used for generation. However, under the
        hood, it uses `temperature` when used by [`BarkModel`]

        This configuration inherit from [`GenerationConfig`] and can be used to control the model generation. Read the
        documentation from [`GenerationConfig`] for more information.

        Args:
            temperature (`float`, *optional*):
                The value used to modulate the next token probabilities.
            max_fine_history_length (`int`, *optional*, defaults to 512):
                Max length of the fine history vector.
            max_fine_input_length (`int`, *optional*, defaults to 1024):
                Max length of fine input vector.
            n_fine_codebooks (`int`, *optional*, defaults to 8):
                Number of codebooks used.
        """
        super().__init__(temperature=temperature)
        self.max_fine_history_length = max_fine_history_length
        self.max_fine_input_length = max_fine_input_length
        self.n_fine_codebooks = n_fine_codebooks

    def validate(self, **kwargs):
        """
        Overrides GenerationConfig.validate because BarkFineGenerationConfig don't use any parameters outside
        temperature.
        """
        pass