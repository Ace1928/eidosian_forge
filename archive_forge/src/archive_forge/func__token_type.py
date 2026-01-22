import collections
import inspect
import re
import warnings
from functools import partial
from typing import Any, Callable, Dict, List, Tuple, Type, Union
from sphinx.application import Sphinx
from sphinx.config import Config as SphinxConfig
from sphinx.deprecation import RemovedInSphinx60Warning
from sphinx.locale import _, __
from sphinx.util import logging
from sphinx.util.inspect import stringify_annotation
from sphinx.util.typing import get_type_hints
def _token_type(token: str, location: str=None) -> str:

    def is_numeric(token):
        try:
            complex(token)
        except ValueError:
            return False
        else:
            return True
    if token.startswith(' ') or token.endswith(' '):
        type_ = 'delimiter'
    elif is_numeric(token) or (token.startswith('{') and token.endswith('}')) or (token.startswith('"') and token.endswith('"')) or (token.startswith("'") and token.endswith("'")):
        type_ = 'literal'
    elif token.startswith('{'):
        logger.warning(__('invalid value set (missing closing brace): %s'), token, location=location)
        type_ = 'literal'
    elif token.endswith('}'):
        logger.warning(__('invalid value set (missing opening brace): %s'), token, location=location)
        type_ = 'literal'
    elif token.startswith("'") or token.startswith('"'):
        logger.warning(__('malformed string literal (missing closing quote): %s'), token, location=location)
        type_ = 'literal'
    elif token.endswith("'") or token.endswith('"'):
        logger.warning(__('malformed string literal (missing opening quote): %s'), token, location=location)
        type_ = 'literal'
    elif token in ('optional', 'default'):
        type_ = 'control'
    elif _xref_regex.match(token):
        type_ = 'reference'
    else:
        type_ = 'obj'
    return type_