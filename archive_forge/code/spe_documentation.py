from datetime import datetime
import logging
import os
from typing import (
import warnings
import numpy as np
from ..core.request import Request, IOMode, InitializationError
from ..core.v3_plugin_api import PluginV3, ImageProperties
Get information from SPE file header

        Parameters
        ----------
        spec
            Maps header entry name to its location, data type description and
            optionally number of entries. See :py:attr:`Spec.basic` and
            :py:attr:`Spec.metadata`.
        char_encoding
            String character encoding

        Returns
        -------
        Dict mapping header entry name to its value
        