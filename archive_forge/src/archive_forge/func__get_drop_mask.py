import warnings
from typing import Callable, List, Optional, Sequence, Tuple, cast
from thinc.api import Model, Ops, registry
from thinc.initializers import glorot_uniform_init
from thinc.types import Floats1d, Floats2d, Ints1d, Ragged
from thinc.util import partial
from ..attrs import ORTH
from ..errors import Errors, Warnings
from ..tokens import Doc
from ..vectors import Mode, Vectors
from ..vocab import Vocab
def _get_drop_mask(ops: Ops, nO: int, rate: Optional[float]) -> Optional[Floats1d]:
    if rate is not None:
        mask = ops.get_dropout_mask((nO,), rate)
        return mask
    return None