from typing import List, Optional
from thinc.api import Model, Softmax_v2, chain, with_array, zero_init
from thinc.types import Floats2d
from ...tokens import Doc
from ...util import registry
Build a tagger model, using a provided token-to-vector component. The tagger
    model simply adds a linear layer with softmax activation to predict scores
    given the token vectors.

    tok2vec (Model[List[Doc], List[Floats2d]]): The token-to-vector subnetwork.
    nO (int or None): The number of tags to output. Inferred from the data if None.
    