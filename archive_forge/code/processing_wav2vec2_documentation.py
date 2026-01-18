import warnings
from contextlib import contextmanager
from ...processing_utils import ProcessorMixin
from .feature_extraction_wav2vec2 import Wav2Vec2FeatureExtractor
from .tokenization_wav2vec2 import Wav2Vec2CTCTokenizer

        Temporarily sets the tokenizer for processing the input. Useful for encoding the labels when fine-tuning
        Wav2Vec2.
        