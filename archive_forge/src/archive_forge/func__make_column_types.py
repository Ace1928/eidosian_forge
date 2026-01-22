import base64
import binascii
import codecs
import datetime
import hashlib
import json
import logging
import os
import pprint
from decimal import Decimal
from typing import Optional
import wandb
from wandb import util
from wandb.sdk.lib import filesystem
from .sdk.data_types import _dtypes
from .sdk.data_types._private import MEDIA_TMP
from .sdk.data_types.base_types.media import (
from .sdk.data_types.base_types.wb_value import WBValue
from .sdk.data_types.helper_types.bounding_boxes_2d import BoundingBoxes2D
from .sdk.data_types.helper_types.classes import Classes
from .sdk.data_types.helper_types.image_mask import ImageMask
from .sdk.data_types.histogram import Histogram
from .sdk.data_types.html import Html
from .sdk.data_types.image import Image
from .sdk.data_types.molecule import Molecule
from .sdk.data_types.object_3d import Object3D
from .sdk.data_types.plotly import Plotly
from .sdk.data_types.saved_model import _SavedModel
from .sdk.data_types.trace_tree import WBTraceTree
from .sdk.data_types.video import Video
from .sdk.lib import runid
def _make_column_types(self, dtype=None, optional=True):
    if dtype is None:
        dtype = _dtypes.UnknownType()
    if optional.__class__ != list:
        optional = [optional for _ in range(len(self.columns))]
    if dtype.__class__ != list:
        dtype = [dtype for _ in range(len(self.columns))]
    self._column_types = _dtypes.TypedDictType({})
    for col_name, opt, dt in zip(self.columns, optional, dtype):
        self.cast(col_name, dt, opt)