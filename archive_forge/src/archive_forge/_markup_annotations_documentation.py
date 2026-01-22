import sys
from abc import ABC
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union
from ..generic import ArrayObject, DictionaryObject
from ..generic._base import (
from ..generic._fit import DEFAULT_FIT, Fit
from ..generic._rectangle import RectangleObject
from ..generic._utils import hex_to_rgb
from ._base import NO_FLAGS, AnnotationDictionary
A FreeText annotation