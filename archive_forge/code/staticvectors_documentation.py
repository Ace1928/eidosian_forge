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
Embed Doc objects with their vocab's vectors table, applying a learned
    linear projection to control the dimensionality. If a dropout rate is
    specified, the dropout is applied per dimension over the whole batch.
    