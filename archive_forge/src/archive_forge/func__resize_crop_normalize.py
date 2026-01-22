from os import PathLike
from typing import Dict, List, Union
from PIL.Image import Image, BICUBIC
from tokenizers import Tokenizer
import numpy as np
def _resize_crop_normalize(self, image: Image):
    width, height = image.size
    if width < height:
        width = self._image_size
        height = int(height / width * self._image_size)
    else:
        width = int(width / height * self._image_size)
        height = self._image_size
    image = image.resize((width, height), resample=BICUBIC)
    left = (width - self._image_size) / 2
    top = (height - self._image_size) / 2
    right = (width + self._image_size) / 2
    bottom = (height + self._image_size) / 2
    image = image.convert('RGB').crop((left, top, right, bottom))
    image = (np.array(image).astype(np.float32) / 255.0 - self.image_mean) / self.image_std
    return np.transpose(image, (2, 0, 1))