import os
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, Type, Union
from .. import _dtypes
from ..base_types.media import Media
Classes is holds class metadata intended to be used in concert with other objects when visualizing artifacts.

        Args:
            class_set (list): list of dicts in the form of {"id":int|str, "name":str}
        