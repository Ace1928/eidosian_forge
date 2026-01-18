from typing import Any, Callable, Dict, Iterable, Optional
from spacy.scorer import PRFScore, ROCAUCScore
from spacy.tokens import Doc
from spacy.training import Example
from spacy.util import SimpleFrozenList
def textcat_multilabel_score_v1(examples: Iterable[Example], **kwargs) -> Dict[str, Any]:
    return score_cats_v1(examples, 'cats', multi_label=True, **kwargs)