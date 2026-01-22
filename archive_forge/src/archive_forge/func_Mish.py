from typing import Callable, Optional, Tuple, cast
from ..config import registry
from ..initializers import glorot_uniform_init, zero_init
from ..model import Model
from ..types import Floats1d, Floats2d
from ..util import get_width, partial
from .chain import chain
from .dropout import Dropout
from .layernorm import LayerNorm
@registry.layers('Mish.v1')
def Mish(nO: Optional[int]=None, nI: Optional[int]=None, *, init_W: Optional[Callable]=None, init_b: Optional[Callable]=None, dropout: Optional[float]=None, normalize: bool=False) -> Model[InT, OutT]:
    """Dense layer with mish activation.
    https://arxiv.org/pdf/1908.08681.pdf
    """
    if init_W is None:
        init_W = glorot_uniform_init
    if init_b is None:
        init_b = zero_init
    model: Model[InT, OutT] = Model('mish', forward, init=partial(init, init_W, init_b), dims={'nO': nO, 'nI': nI}, params={'W': None, 'b': None})
    if normalize:
        model = chain(model, cast(Model[InT, OutT], LayerNorm(nI=nO)))
    if dropout is not None:
        model = chain(model, cast(Model[InT, OutT], Dropout(dropout)))
    return model