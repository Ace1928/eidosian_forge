import warnings
from itertools import product
import numpy as np
import pytest
from ..filebasedimages import FileBasedHeader, FileBasedImage, SerializableImage
from .test_image_api import GenericImageAPI, SerializeMixin
class H(FileBasedHeader):

    def __init__(self, seq=None):
        if seq is None:
            seq = []
        self.a_list = list(seq)