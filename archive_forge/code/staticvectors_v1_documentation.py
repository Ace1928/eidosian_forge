from typing import List, Tuple, Callable, Optional, cast
from thinc.util import partial
from thinc.types import Ragged, Floats2d, Floats1d
from thinc.api import Model, Ops
from thinc.initializers import glorot_uniform_init
from spacy.tokens import Doc
from spacy.errors import Errors
from spacy.util import registry
Embed Doc objects with their vocab's vectors table, applying a learned
    linear projection to control the dimensionality. If a dropout rate is
    specified, the dropout is applied per dimension over the whole batch.
    