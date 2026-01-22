import copy
from typing import Dict
from ...generation.configuration_utils import GenerationConfig
from ...utils import logging
class BarkGenerationConfig(GenerationConfig):
    model_type = 'bark'
    is_composition = True

    def __init__(self, semantic_config: Dict=None, coarse_acoustics_config: Dict=None, fine_acoustics_config: Dict=None, sample_rate=24000, codebook_size=1024, **kwargs):
        """Class that holds a generation configuration for [`BarkModel`].

        The [`BarkModel`] does not have a `generate` method, but uses this class to generate speeches with a nested
        [`BarkGenerationConfig`] which uses [`BarkSemanticGenerationConfig`], [`BarkCoarseGenerationConfig`],
        [`BarkFineGenerationConfig`].

        This configuration inherit from [`GenerationConfig`] and can be used to control the model generation. Read the
        documentation from [`GenerationConfig`] for more information.

        Args:
            semantic_config (`Dict`, *optional*):
                Semantic generation configuration.
            coarse_acoustics_config (`Dict`, *optional*):
                Coarse generation configuration.
            fine_acoustics_config (`Dict`, *optional*):
                Fine generation configuration.
            sample_rate (`int`, *optional*, defaults to 24_000):
                Sample rate.
            codebook_size (`int`, *optional*, defaults to 1024):
                Vector length for each codebook.
        """
        if semantic_config is None:
            semantic_config = {}
            logger.info('semantic_config is None. initializing the semantic model with default values.')
        if coarse_acoustics_config is None:
            coarse_acoustics_config = {}
            logger.info('coarse_acoustics_config is None. initializing the coarse model with default values.')
        if fine_acoustics_config is None:
            fine_acoustics_config = {}
            logger.info('fine_acoustics_config is None. initializing the fine model with default values.')
        self.semantic_config = BarkSemanticGenerationConfig(**semantic_config)
        self.coarse_acoustics_config = BarkCoarseGenerationConfig(**coarse_acoustics_config)
        self.fine_acoustics_config = BarkFineGenerationConfig(**fine_acoustics_config)
        self.sample_rate = sample_rate
        self.codebook_size = codebook_size

    @classmethod
    def from_sub_model_configs(cls, semantic_config: BarkSemanticGenerationConfig, coarse_acoustics_config: BarkCoarseGenerationConfig, fine_acoustics_config: BarkFineGenerationConfig, **kwargs):
        """
        Instantiate a [`BarkGenerationConfig`] (or a derived class) from bark sub-models generation configuration.

        Returns:
            [`BarkGenerationConfig`]: An instance of a configuration object
        """
        return cls(semantic_config=semantic_config.to_dict(), coarse_acoustics_config=coarse_acoustics_config.to_dict(), fine_acoustics_config=fine_acoustics_config.to_dict(), **kwargs)

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output['semantic_config'] = self.semantic_config.to_dict()
        output['coarse_acoustics_config'] = self.coarse_acoustics_config.to_dict()
        output['fine_acoustics_config'] = self.fine_acoustics_config.to_dict()
        output['model_type'] = self.__class__.model_type
        return output