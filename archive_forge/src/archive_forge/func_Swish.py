from typing import Callable, Optional, Tuple, cast
from ..config import registry
from ..initializers import he_normal_init, zero_init
from ..model import Model
from ..types import Floats1d, Floats2d
from ..util import get_width, partial
from .chain import chain
from .dropout import Dropout
from .layernorm import LayerNorm
@registry.layers('Swish.v1')
def Swish(nO: Optional[int]=None, nI: Optional[int]=None, *, init_W: Optional[Callable]=None, init_b: Optional[Callable]=None, dropout: Optional[float]=None, normalize: bool=False) -> Model[Floats2d, Floats2d]:
    if init_W is None:
        init_W = he_normal_init
    if init_b is None:
        init_b = zero_init
    model: Model[Floats2d, Floats2d] = Model('swish', forward, init=partial(init, init_W, init_b), dims={'nO': nO, 'nI': nI}, params={'W': None, 'b': None})
    if normalize:
        model = chain(model, LayerNorm(nI=nO))
    if dropout is not None:
        model = chain(model, cast(Model[Floats2d, Floats2d], Dropout(dropout)))
    return model