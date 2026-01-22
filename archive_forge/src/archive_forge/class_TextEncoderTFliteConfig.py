from ...utils import DummyTextInputGenerator, DummyVisionInputGenerator, logging
from .base import TFLiteConfig
class TextEncoderTFliteConfig(TFLiteConfig):
    """
    Handles encoder-based text architectures.
    """
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator,)
    MANDATORY_AXES = ('batch_size', 'sequence_length', ('multiple-choice', 'num_choices'))