from typing import Dict, Iterable, Optional, Union
import numpy as np
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import resize, to_channel_dimension_format, to_pil_image
from ...image_utils import (
from ...utils import TensorType, is_pytesseract_available, is_vision_available, logging, requires_backends
def apply_tesseract(image: np.ndarray, lang: Optional[str], tesseract_config: Optional[str], input_data_format: Optional[Union[ChannelDimension, str]]=None):
    """Applies Tesseract OCR on a document image, and returns recognized words + normalized bounding boxes."""
    pil_image = to_pil_image(image, input_data_format=input_data_format)
    image_width, image_height = pil_image.size
    data = pytesseract.image_to_data(pil_image, lang=lang, output_type='dict', config=tesseract_config)
    words, left, top, width, height = (data['text'], data['left'], data['top'], data['width'], data['height'])
    irrelevant_indices = [idx for idx, word in enumerate(words) if not word.strip()]
    words = [word for idx, word in enumerate(words) if idx not in irrelevant_indices]
    left = [coord for idx, coord in enumerate(left) if idx not in irrelevant_indices]
    top = [coord for idx, coord in enumerate(top) if idx not in irrelevant_indices]
    width = [coord for idx, coord in enumerate(width) if idx not in irrelevant_indices]
    height = [coord for idx, coord in enumerate(height) if idx not in irrelevant_indices]
    actual_boxes = []
    for x, y, w, h in zip(left, top, width, height):
        actual_box = [x, y, x + w, y + h]
        actual_boxes.append(actual_box)
    normalized_boxes = []
    for box in actual_boxes:
        normalized_boxes.append(normalize_box(box, image_width, image_height))
    assert len(words) == len(normalized_boxes), 'Not as many words as there are bounding boxes'
    return (words, normalized_boxes)