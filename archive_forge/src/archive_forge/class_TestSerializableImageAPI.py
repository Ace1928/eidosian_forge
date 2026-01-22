import warnings
from itertools import product
import numpy as np
import pytest
from ..filebasedimages import FileBasedHeader, FileBasedImage, SerializableImage
from .test_image_api import GenericImageAPI, SerializeMixin
class TestSerializableImageAPI(TestFBImageAPI, SerializeMixin):
    image_maker = SerializableNumpyImage

    @staticmethod
    def _header_eq(header_a, header_b):
        """FileBasedHeader is an abstract class, so __eq__ is undefined.
        Checking for the same header type is sufficient, here."""
        return type(header_a) == type(header_b) == FileBasedHeader