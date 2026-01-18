import codecs
import os
from typing import TYPE_CHECKING, Sequence, Type, Union
from wandb import util
from wandb.sdk.lib import runid
from ._private import MEDIA_TMP
from .base_types.media import Media, _numpy_arrays_to_lists
from .base_types.wb_value import WBValue
from .image import Image
Wandb class for plotly plots.

    Arguments:
        val: matplotlib or plotly figure
    