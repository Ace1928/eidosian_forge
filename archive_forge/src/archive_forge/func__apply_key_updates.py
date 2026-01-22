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
def _apply_key_updates(self, only_last=False):
    """Appropriately wraps the underlying data in special Key classes.

        Arguments:
            only_last: only apply the updates to the last row (used for performance when
            the caller knows that the only new data is the last row and no updates were
            applied to the column types)
        """
    c_types = self._column_types.params['type_map']

    def update_row(row_ndx):
        for fk_col in self._fk_cols:
            col_ndx = self.columns.index(fk_col)
            if isinstance(c_types[fk_col], _ForeignKeyType) and (not isinstance(self.data[row_ndx][col_ndx], _TableKey)):
                self.data[row_ndx][col_ndx] = _TableKey(self.data[row_ndx][col_ndx])
                self.data[row_ndx][col_ndx].set_table(c_types[fk_col].params['table'], c_types[fk_col].params['col_name'])
            elif isinstance(c_types[fk_col], _ForeignIndexType) and (not isinstance(self.data[row_ndx][col_ndx], _TableIndex)):
                self.data[row_ndx][col_ndx] = _TableIndex(self.data[row_ndx][col_ndx])
                self.data[row_ndx][col_ndx].set_table(c_types[fk_col].params['table'])
        if self._pk_col is not None:
            col_ndx = self.columns.index(self._pk_col)
            self.data[row_ndx][col_ndx] = _TableKey(self.data[row_ndx][col_ndx])
            self.data[row_ndx][col_ndx].set_table(self, self._pk_col)
    if only_last:
        update_row(len(self.data) - 1)
    else:
        for row_ndx in range(len(self.data)):
            update_row(row_ndx)