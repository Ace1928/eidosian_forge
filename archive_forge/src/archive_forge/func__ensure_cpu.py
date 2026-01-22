from typing import Callable, List, Tuple
from thinc.api import Model, to_numpy
from thinc.types import Ints1d, Ragged
from ..util import registry
def _ensure_cpu(spans: Ragged, lengths: Ints1d) -> Tuple[Ragged, Ints1d]:
    return (Ragged(to_numpy(spans.dataXd), to_numpy(spans.lengths)), to_numpy(lengths))