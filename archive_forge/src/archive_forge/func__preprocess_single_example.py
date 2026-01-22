import copy
import math
import re
from typing import List, Optional, Tuple, Union
from ...image_processing_utils import BatchFeature
from ...image_utils import ImageInput, is_batched
from ...processing_utils import ProcessorMixin
from ...tokenization_utils import AddedToken
from ...tokenization_utils_base import BatchEncoding, PaddingStrategy, TextInput, TruncationStrategy
from ...utils import TensorType
def _preprocess_single_example(self, text, image, bboxes, img_info_tokens):
    text = text.strip()
    if image is not None:
        text = f'{img_info_tokens} {text}'
    text = self._insert_patch_index_tokens(text, bboxes)
    return text