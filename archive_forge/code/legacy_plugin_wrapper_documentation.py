from pathlib import Path
import numpy as np
from ..config import known_extensions
from .request import InitializationError, IOMode
from .v3_plugin_api import ImageProperties, PluginV3
Format-Specific ndimage metadata.

        Parameters
        ----------
        index : int
            The index of the ndimage to read. If the index is out of bounds a
            ``ValueError`` is raised. If ``None``, global metadata is returned.
        exclude_applied : bool
            This parameter exists for compatibility and has no effect. Legacy
            plugins always report all metadata they find.

        Returns
        -------
        metadata : dict
            A dictionary filled with format-specific metadata fields and their
            values.

        