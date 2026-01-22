from typing import Optional, List
from thinc.types import Floats2d
from thinc.api import Model, with_cpu
from spacy.attrs import ID, ORTH, PREFIX, SUFFIX, SHAPE, LOWER
from spacy.util import registry
from spacy.tokens import Doc
def TextCatBOW_v1(exclusive_classes: bool, ngram_size: int, no_output_layer: bool, nO: Optional[int]=None) -> Model[List[Doc], Floats2d]:
    chain = registry.get('layers', 'chain.v1')
    Logistic = registry.get('layers', 'Logistic.v1')
    SparseLinear = registry.get('layers', 'SparseLinear.v1')
    softmax_activation = registry.get('layers', 'softmax_activation.v1')
    extract_ngrams = registry.get('layers', 'spacy.extract_ngrams.v1')
    with Model.define_operators({'>>': chain}):
        sparse_linear = SparseLinear(nO)
        model = extract_ngrams(ngram_size, attr=ORTH) >> sparse_linear
        model = with_cpu(model, model.ops)
        if not no_output_layer:
            output_layer = softmax_activation() if exclusive_classes else Logistic()
            model = model >> with_cpu(output_layer, output_layer.ops)
    model.set_ref('output_layer', sparse_linear)
    model.attrs['multi_label'] = not exclusive_classes
    return model