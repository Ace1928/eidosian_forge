from os import PathLike
from typing import Dict, List, Union
import torch
from PIL.Image import Image
from tokenizers import Tokenizer
from torch import Tensor
from torchvision.transforms import (
Transforms one or more Pillow images into Torch Tensors.

        :param images: image or list of images to preprocess
        