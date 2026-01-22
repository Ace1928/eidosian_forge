from typing import List, Tuple, Callable, Optional, cast
from thinc.util import partial
from thinc.types import Ragged, Floats2d, Floats1d
from thinc.api import Model, Ops
from thinc.initializers import glorot_uniform_init
from spacy.tokens import Doc
from spacy.errors import Errors
from spacy.util import registry
def StaticVectors_v1(nO: Optional[int]=None, nM: Optional[int]=None, *, dropout: Optional[float]=None, init_W: Callable=glorot_uniform_init, key_attr: str='ORTH') -> Model[List[Doc], Ragged]:
    """Embed Doc objects with their vocab's vectors table, applying a learned
    linear projection to control the dimensionality. If a dropout rate is
    specified, the dropout is applied per dimension over the whole batch.
    """
    return Model('static_vectors', forward, init=partial(init, init_W), params={'W': None}, attrs={'key_attr': key_attr, 'dropout_rate': dropout}, dims={'nO': nO, 'nM': nM})