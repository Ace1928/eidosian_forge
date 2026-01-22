from http://www.nitrc.org/projects/gifti/
from __future__ import annotations
import base64
import sys
import warnings
from copy import copy
from typing import Type, cast
import numpy as np
from .. import xmlutils as xml
from ..caret import CaretMetaData
from ..deprecated import deprecate_with_version
from ..filebasedimages import SerializableImage
from ..nifti1 import data_type_codes, intent_codes, xform_codes
from .util import KIND2FMT, array_index_order_codes, gifti_encoding_codes, gifti_endian_codes
from .parse_gifti_fast import GiftiImageParser
class GiftiLabel(xml.XmlSerializable):
    """Gifti label: association of integer key with optional RGBA values

    Quotes are from the gifti spec dated 2011-01-14.

    Attributes
    ----------
    key : int
        (From the spec): "This required attribute contains a non-negative
        integer value. If a DataArray's Intent is NIFTI_INTENT_LABEL and a
        value in the DataArray is 'X', its corresponding label is the label
        with the Key attribute containing the value 'X'. In early versions of
        the GIFTI file format, the attribute Index was used instead of Key. If
        an Index attribute is encountered, it should be processed like the Key
        attribute."
    red : None or float
        Optional value for red.
    green : None or float
        Optional value for green.
    blue : None or float
        Optional value for blue.
    alpha : None or float
        Optional value for alpha.

    Notes
    -----
    freesurfer examples seem not to conform to datatype "NIFTI_TYPE_RGBA32"
    because they are floats, not 4 8-bit integers.
    """

    def __init__(self, key=0, red=None, green=None, blue=None, alpha=None):
        self.key = key
        self.red = red
        self.green = green
        self.blue = blue
        self.alpha = alpha

    def __repr__(self):
        chars = 255 * np.array([self.red or 0, self.green or 0, self.blue or 0, self.alpha or 0])
        r, g, b, a = chars.astype('u1')
        return f'<GiftiLabel {self.key}="#{r:02x}{g:02x}{b:02x}{a:02x}">'

    @property
    def rgba(self):
        """Returns RGBA as tuple"""
        return (self.red, self.green, self.blue, self.alpha)

    @rgba.setter
    def rgba(self, rgba):
        """Set RGBA via sequence

        Parameters
        ----------
        rgba : length 4 sequence
            Sequence containing values for red, green, blue, alpha.
        """
        if len(rgba) != 4:
            raise ValueError('rgba must be length 4.')
        self.red, self.green, self.blue, self.alpha = rgba