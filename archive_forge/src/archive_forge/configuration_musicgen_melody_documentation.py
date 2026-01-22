from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..auto.configuration_auto import AutoConfig
from ..deprecated._archive_maps import MUSICGEN_MELODY_PRETRAINED_CONFIG_ARCHIVE_MAP  # noqa: F401, E402

        Instantiate a [`MusicgenMelodyConfig`] (or a derived class) from text encoder, audio encoder and decoder
        configurations.

        Returns:
            [`MusicgenMelodyConfig`]: An instance of a configuration object
        