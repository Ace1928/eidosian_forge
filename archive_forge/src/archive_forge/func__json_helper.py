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
def _json_helper(val, artifact):
    if isinstance(val, WBValue):
        return val.to_json(artifact)
    elif val.__class__ == dict:
        res = {}
        for key in val:
            res[key] = _json_helper(val[key], artifact)
        return res
    if hasattr(val, 'tolist'):
        py_val = val.tolist()
        if val.__class__.__name__ == 'datetime64' and isinstance(py_val, int):
            return _json_helper(py_val / int(1000000.0), artifact)
        return _json_helper(py_val, artifact)
    elif hasattr(val, 'item'):
        return _json_helper(val.item(), artifact)
    if isinstance(val, datetime.datetime):
        if val.tzinfo is None:
            val = datetime.datetime(val.year, val.month, val.day, val.hour, val.minute, val.second, val.microsecond, tzinfo=datetime.timezone.utc)
        return int(val.timestamp() * 1000)
    elif isinstance(val, datetime.date):
        return int(datetime.datetime(val.year, val.month, val.day, tzinfo=datetime.timezone.utc).timestamp() * 1000)
    elif isinstance(val, (list, tuple)):
        return [_json_helper(i, artifact) for i in val]
    elif isinstance(val, Decimal):
        return float(val)
    else:
        return util.json_friendly(val)[0]