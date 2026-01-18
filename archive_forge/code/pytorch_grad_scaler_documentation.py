from typing import Dict, Iterable, List, Union, cast
from ..compat import has_torch_amp, torch
from ..util import is_torch_array

        Update the scale factor and clear information about infinities.

        This method should be called after each optimization step.
        