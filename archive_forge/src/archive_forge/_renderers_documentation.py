import textwrap
from copy import copy
import os
from packaging.version import Version
from plotly import optional_imports
from plotly.io._base_renderers import (
from plotly.io._utils import validate_coerce_fig_to_dict

        Perform external rendering for each ExternalRenderer specified
        in either the default renderer string, or in the supplied
        renderers_string argument.

        Note that this method skips any renderers that are not subclasses
        of ExternalRenderer.

        Parameters
        ----------
        fig_dict: dict
            Figure dictionary
        renderers_string: str or None (default None)
            Renderer string to process rather than the current default
            renderer string

        Returns
        -------
        None
        