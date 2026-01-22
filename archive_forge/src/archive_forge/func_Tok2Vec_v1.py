from typing import List, Optional, Union
from thinc.api import Model
from thinc.types import Floats2d
from spacy.attrs import intify_attr
from spacy.errors import Errors
from spacy.tokens import Doc
from spacy.util import registry
def Tok2Vec_v1(embed: Model[List[Doc], List[Floats2d]], encode: Model[Floats2d, Floats2d]) -> Model[List[Doc], List[Floats2d]]:
    """Construct a tok2vec model out of embedding and encoding subnetworks.
    See https://explosion.ai/blog/deep-learning-formula-nlp

    embed (Model[List[Doc], List[Floats2d]]): Embed tokens into context-independent
        word vector representations.
    encode (Model[List[Floats2d], List[Floats2d]]): Encode context into the
        embeddings, using an architecture such as a CNN, BiLSTM or transformer.
    """
    chain = registry.get('layers', 'chain.v1')
    with_array = registry.get('layers', 'with_array.v1')
    receptive_field = encode.attrs.get('receptive_field', 0)
    tok2vec = chain(embed, with_array(encode, pad=receptive_field))
    tok2vec.set_dim('nO', encode.get_dim('nO'))
    tok2vec.set_ref('embed', embed)
    tok2vec.set_ref('encode', encode)
    return tok2vec