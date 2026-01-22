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
def _update_keys(self, force_last=False):
    """Updates the known key-like columns based on current column types.

        If the state has been updated since the last update, wraps the data
        appropriately in the Key classes.

        Arguments:
            force_last: (bool) Wraps the last column of data even if there
                are no key updates.
        """
    _pk_col = None
    _fk_cols = set()
    c_types = self._column_types.params['type_map']
    for t in c_types:
        if isinstance(c_types[t], _PrimaryKeyType):
            _pk_col = t
        elif isinstance(c_types[t], _ForeignKeyType) or isinstance(c_types[t], _ForeignIndexType):
            _fk_cols.add(t)
    has_update = _pk_col != self._pk_col or _fk_cols != self._fk_cols
    if has_update:
        if _pk_col is None and self._pk_col is not None:
            raise AssertionError(f'Cannot unset primary key (column {self._pk_col})')
        if len(self._fk_cols - _fk_cols) > 0:
            raise AssertionError('Cannot unset foreign key. Attempted to unset ({})'.format(self._fk_cols - _fk_cols))
        self._pk_col = _pk_col
        self._fk_cols = _fk_cols
    if has_update or force_last:
        self._apply_key_updates(not has_update)