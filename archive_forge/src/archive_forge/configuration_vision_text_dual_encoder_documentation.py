from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..auto.configuration_auto import AutoConfig
from ..chinese_clip.configuration_chinese_clip import ChineseCLIPVisionConfig
from ..clip.configuration_clip import CLIPVisionConfig
from ..siglip.configuration_siglip import SiglipVisionConfig

        Instantiate a [`VisionTextDualEncoderConfig`] (or a derived class) from text model configuration and vision
        model configuration.

        Returns:
            [`VisionTextDualEncoderConfig`]: An instance of a configuration object
        