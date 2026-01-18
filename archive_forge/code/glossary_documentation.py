import warnings
from .errors import Warnings
Get a description for a given POS tag, dependency label or entity type.

    term (str): The term to explain.
    RETURNS (str): The explanation, or `None` if not found in the glossary.

    EXAMPLE:
        >>> spacy.explain(u'NORP')
        >>> doc = nlp(u'Hello world')
        >>> print([w.text, w.tag_, spacy.explain(w.tag_) for w in doc])
    