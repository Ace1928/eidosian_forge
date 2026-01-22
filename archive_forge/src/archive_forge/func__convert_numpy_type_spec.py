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
def _convert_numpy_type_spec(_type: str, location: str=None, translations: dict={}) -> str:

    def convert_obj(obj, translations, default_translation):
        translation = translations.get(obj, obj)
        if translation in _SINGLETONS and default_translation == ':class:`%s`':
            default_translation = ':obj:`%s`'
        elif translation == '...' and default_translation == ':class:`%s`':
            default_translation = ':obj:`%s <Ellipsis>`'
        if _xref_regex.match(translation) is None:
            translation = default_translation % translation
        return translation
    tokens = _tokenize_type_spec(_type)
    combined_tokens = _recombine_set_tokens(tokens)
    types = [(token, _token_type(token, location)) for token in combined_tokens]
    converters = {'literal': lambda x: '``%s``' % x, 'obj': lambda x: convert_obj(x, translations, ':class:`%s`'), 'control': lambda x: '*%s*' % x, 'delimiter': lambda x: x, 'reference': lambda x: x}
    converted = ''.join((converters.get(type_)(token) for token, type_ in types))
    return converted