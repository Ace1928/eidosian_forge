import warnings
from itertools import product
import numpy as np
import pytest
from ..filebasedimages import FileBasedHeader, FileBasedImage, SerializableImage
from .test_image_api import GenericImageAPI, SerializeMixin
FileBasedHeader is an abstract class, so __eq__ is undefined.
        Checking for the same header type is sufficient, here.