import re
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor, tensor
from typing_extensions import Literal
from torchmetrics.utilities.imports import _NLTK_AVAILABLE
def _ensure_nltk_punkt_is_downloaded() -> None:
    """Check whether `nltk` `punkt` is downloaded.

    If not, try to download if a machine is connected to the internet.

    """
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        try:
            nltk.download('punkt', quiet=True, force=False, halt_on_error=False, raise_on_error=True)
        except ValueError as err:
            raise OSError('`nltk` resource `punkt` is not available on a disk and cannot be downloaded as a machine is not connected to the internet.') from err