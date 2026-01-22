from typing import Callable, Optional, Tuple, cast
from ..config import registry
from ..model import Model
from ..types import Floats2d, Ragged
from ..util import get_width
from .noop import noop
@registry.layers('ParametricAttention.v2')
def ParametricAttention_v2(*, key_transform: Optional[Model[Floats2d, Floats2d]]=None, nO: Optional[int]=None) -> Model[InT, OutT]:
    if key_transform is None:
        key_transform = noop()
    'Weight inputs by similarity to a learned vector'
    return Model('para-attn', forward, init=init, params={'Q': None}, dims={'nO': nO}, refs={KEY_TRANSFORM_REF: key_transform}, layers=[key_transform])