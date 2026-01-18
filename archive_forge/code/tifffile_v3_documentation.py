from io import BytesIO
from typing import Any, Dict, Optional, cast
import warnings
import numpy as np
import tifffile
from ..core.request import URI_BYTES, InitializationError, Request
from ..core.v3_plugin_api import ImageProperties, PluginV3
from ..typing import ArrayLike
Yield pages from a TIFF file.

        This generator walks over the flat index of the pages inside an
        ImageResource and yields them in order.

        Parameters
        ----------
        index : int
            The index of the series to yield pages from. If Ellipsis, walk over
            the file's flat index (and ignore individual series).
        kwargs : Any
            Additional kwargs are passed to TiffPage's ``as_array`` method.

        Yields
        ------
        page : np.ndarray
            A page stored inside the TIFF file.

        