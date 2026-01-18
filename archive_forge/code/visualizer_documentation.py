import itertools
import os
import re
from string import Template
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple
from tokenizers import Encoding, Tokenizer

        For each character in the original text, we emit a tuple representing it's "state":

            * which token_ix it corresponds to
            * which word_ix it corresponds to
            * which annotation_ix it corresponds to

        Args:
            text (:obj:`str`):
                The raw text we want to align to

            annotations (:obj:`List[Annotation]`):
                A (possibly empty) list of annotations

            encoding: (:class:`~tokenizers.Encoding`):
                The encoding returned from the tokenizer

        Returns:
            :obj:`List[CharState]`: A list of CharStates, indicating for each char in the text what
            it's state is
        