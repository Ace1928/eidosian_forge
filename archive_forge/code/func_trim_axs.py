from itertools import product
from math import ceil, floor, sqrt
from typing import Any, Dict, Generator, List, Optional, Sequence, Tuple, Union, no_type_check
import numpy as np
import torch
from torch import Tensor
from torchmetrics.utilities.imports import _LATEX_AVAILABLE, _MATPLOTLIB_AVAILABLE, _SCIENCEPLOT_AVAILABLE
def trim_axs(axs: Union[_AX_TYPE, np.ndarray], nb: int) -> Union[np.ndarray, _AX_TYPE]:
    """Reduce `axs` to `nb` Axes.

    All further Axes are removed from the figure.

    """
    if isinstance(axs, _AX_TYPE):
        return axs
    axs = axs.flat
    for ax in axs[nb:]:
        ax.remove()
    return axs[:nb]