import copy
import json
import os
import re
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union
from packaging import version
from . import __version__
from .dynamic_module_utils import custom_object_save
from .utils import (
@property
def use_return_dict(self) -> bool:
    """
        `bool`: Whether or not return [`~utils.ModelOutput`] instead of tuples.
        """
    return self.return_dict and (not self.torchscript)