import warnings
from typing import Dict, Iterable, Iterator, List, Tuple, Union, cast
from ..errors import Errors, Warnings
from ..tokens import Doc, Span
def _doc_to_biluo_tags_with_partial(doc: Doc) -> List[str]:
    ents = doc_to_biluo_tags(doc, missing='-')
    for i, token in enumerate(doc):
        if token.ent_iob == 2:
            ents[i] = 'O'
    return ents