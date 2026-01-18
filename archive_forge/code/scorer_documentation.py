from collections import defaultdict
from typing import (
import numpy as np
from .errors import Errors
from .morphology import Morphology
from .tokens import Doc, Span, Token
from .training import Example
from .util import SimpleFrozenList, get_lang_class
Returns the UAS, LAS, and LAS per type scores for dependency
        parses.

        examples (Iterable[Example]): Examples to score
        attr (str): The attribute containing the dependency label.
        getter (Callable[[Token, str], Any]): Defaults to getattr. If provided,
            getter(token, attr) should return the value of the attribute for an
            individual token.
        head_attr (str): The attribute containing the head token. Defaults to
            'head'.
        head_getter (Callable[[Token, str], Token]): Defaults to getattr. If provided,
            head_getter(token, attr) should return the value of the head for an
            individual token.
        ignore_labels (Tuple): Labels to ignore while scoring (e.g., punct).
        missing_values (Set[Any]): Attribute values to treat as missing annotation
            in the reference annotation.
        RETURNS (Dict[str, Any]): A dictionary containing the scores:
            attr_uas, attr_las, and attr_las_per_type.

        DOCS: https://spacy.io/api/scorer#score_deps
        