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
def adjust_entity_positions(entity, text):
    """Adjust the positions of the entities in `text` to be relative to the text with special fields removed."""
    entity_name, (start, end) = entity
    adjusted_start = len(re.sub('<.*?>', '', text[:start]))
    adjusted_end = len(re.sub('<.*?>', '', text[:end]))
    adjusted_entity = (entity_name, (adjusted_start, adjusted_end))
    return adjusted_entity