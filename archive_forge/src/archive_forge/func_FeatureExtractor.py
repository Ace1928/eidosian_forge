from typing import Callable, List, Tuple, Union
from thinc.api import Model, registry
from thinc.types import Ints2d
from ..tokens import Doc
@registry.layers('spacy.FeatureExtractor.v1')
def FeatureExtractor(columns: List[Union[int, str]]) -> Model[List[Doc], List[Ints2d]]:
    return Model('extract_features', forward, attrs={'columns': columns})