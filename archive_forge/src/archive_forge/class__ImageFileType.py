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
class _ImageFileType(_dtypes.Type):
    name = 'image-file'
    legacy_names = ['wandb.Image']
    types = [Image]

    def __init__(self, box_layers=None, box_score_keys=None, mask_layers=None, class_map=None, **kwargs):
        box_layers = box_layers or {}
        box_score_keys = box_score_keys or []
        mask_layers = mask_layers or {}
        class_map = class_map or {}
        if isinstance(box_layers, _dtypes.ConstType):
            box_layers = box_layers._params['val']
        if not isinstance(box_layers, dict):
            raise TypeError('box_layers must be a dict')
        else:
            box_layers = _dtypes.ConstType({layer_key: set(box_layers[layer_key]) for layer_key in box_layers})
        if isinstance(mask_layers, _dtypes.ConstType):
            mask_layers = mask_layers._params['val']
        if not isinstance(mask_layers, dict):
            raise TypeError('mask_layers must be a dict')
        else:
            mask_layers = _dtypes.ConstType({layer_key: set(mask_layers[layer_key]) for layer_key in mask_layers})
        if isinstance(box_score_keys, _dtypes.ConstType):
            box_score_keys = box_score_keys._params['val']
        if not isinstance(box_score_keys, list) and (not isinstance(box_score_keys, set)):
            raise TypeError('box_score_keys must be a list or a set')
        else:
            box_score_keys = _dtypes.ConstType(set(box_score_keys))
        if isinstance(class_map, _dtypes.ConstType):
            class_map = class_map._params['val']
        if not isinstance(class_map, dict):
            raise TypeError('class_map must be a dict')
        else:
            class_map = _dtypes.ConstType(class_map)
        self.params.update({'box_layers': box_layers, 'box_score_keys': box_score_keys, 'mask_layers': mask_layers, 'class_map': class_map})

    def assign_type(self, wb_type=None):
        if isinstance(wb_type, _ImageFileType):
            box_layers_self = self.params['box_layers'].params['val'] or {}
            box_score_keys_self = self.params['box_score_keys'].params['val'] or []
            mask_layers_self = self.params['mask_layers'].params['val'] or {}
            class_map_self = self.params['class_map'].params['val'] or {}
            box_layers_other = wb_type.params['box_layers'].params['val'] or {}
            box_score_keys_other = wb_type.params['box_score_keys'].params['val'] or []
            mask_layers_other = wb_type.params['mask_layers'].params['val'] or {}
            class_map_other = wb_type.params['class_map'].params['val'] or {}
            box_layers = {str(key): set(list(box_layers_self.get(key, [])) + list(box_layers_other.get(key, []))) for key in set(list(box_layers_self.keys()) + list(box_layers_other.keys()))}
            mask_layers = {str(key): set(list(mask_layers_self.get(key, [])) + list(mask_layers_other.get(key, []))) for key in set(list(mask_layers_self.keys()) + list(mask_layers_other.keys()))}
            box_score_keys = set(list(box_score_keys_self) + list(box_score_keys_other))
            class_map = {str(key): class_map_self.get(key, class_map_other.get(key, None)) for key in set(list(class_map_self.keys()) + list(class_map_other.keys()))}
            return _ImageFileType(box_layers, box_score_keys, mask_layers, class_map)
        return _dtypes.InvalidType()

    @classmethod
    def from_obj(cls, py_obj):
        if not isinstance(py_obj, Image):
            raise TypeError('py_obj must be a wandb.Image')
        else:
            if hasattr(py_obj, '_boxes') and py_obj._boxes:
                box_layers = {str(key): set(py_obj._boxes[key]._class_labels.keys()) for key in py_obj._boxes.keys()}
                box_score_keys = {key for val in py_obj._boxes.values() for box in val._val for key in box.get('scores', {}).keys()}
            else:
                box_layers = {}
                box_score_keys = set()
            if hasattr(py_obj, '_masks') and py_obj._masks:
                mask_layers = {str(key): set(py_obj._masks[key]._val['class_labels'].keys() if hasattr(py_obj._masks[key], '_val') else []) for key in py_obj._masks.keys()}
            else:
                mask_layers = {}
            if hasattr(py_obj, '_classes') and py_obj._classes:
                class_set = {str(item['id']): item['name'] for item in py_obj._classes._class_set}
            else:
                class_set = {}
            return cls(box_layers, box_score_keys, mask_layers, class_set)