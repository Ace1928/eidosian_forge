import warnings
from ...processing_utils import ProcessorMixin
from ..seamless_m4t.feature_extraction_seamless_m4t import SeamlessM4TFeatureExtractor
from ..wav2vec2.tokenization_wav2vec2 import Wav2Vec2CTCTokenizer

        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.decode`]. Please refer
        to the docstring of this method for more information.
        