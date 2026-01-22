from typing import Callable, Optional, Tuple, cast
from ..config import registry
from ..initializers import glorot_uniform_init, zero_init
from ..model import Model
from ..types import Floats1d, Floats2d
from ..util import get_width, partial
from .chain import chain
from .dropout import Dropout
from .layernorm import LayerNorm
@registry.layers('ClippedLinear.v1')
def ClippedLinear(nO: Optional[int]=None, nI: Optional[int]=None, *, init_W: Optional[Callable]=None, init_b: Optional[Callable]=None, dropout: Optional[float]=None, normalize: bool=False, slope: float=1.0, offset: float=0.0, min_val: float=0.0, max_val: float=1.0) -> Model[Floats2d, Floats2d]:
    if init_W is None:
        init_W = glorot_uniform_init
    if init_b is None:
        init_b = zero_init
    model_attrs = {'slope': slope, 'offset': offset, 'min_val': min_val, 'max_val': max_val}
    model: Model[Floats2d, Floats2d] = Model('clipped_linear', forward=forward, init=partial(init, init_W, init_b), dims={'nO': nO, 'nI': nI}, params={'W': None, 'b': None}, attrs=model_attrs)
    if normalize:
        model = chain(model, LayerNorm(nI=nO))
    if dropout is not None:
        model = chain(model, cast(Model[Floats2d, Floats2d], Dropout(dropout)))
    return model