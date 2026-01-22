import io
import json
import logging
import os
import re
from contextlib import contextmanager
from textwrap import indent, wrap
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union, cast
from .fastjsonschema_exceptions import JsonSchemaValueException
class _SummaryWriter:
    _IGNORE = {'description', 'default', 'title', 'examples'}

    def __init__(self, jargon: Optional[Dict[str, str]]=None):
        self.jargon: Dict[str, str] = jargon or {}
        self._terms = {'anyOf': 'at least one of the following', 'oneOf': 'exactly one of the following', 'allOf': 'all of the following', 'not': '(*NOT* the following)', 'prefixItems': f'{self._jargon('items')} (in order)', 'items': 'items', 'contains': 'contains at least one of', 'propertyNames': f'non-predefined acceptable {self._jargon('property names')}', 'patternProperties': f'{self._jargon('properties')} named via pattern', 'const': 'predefined value', 'enum': 'one of'}
        self._guess_inline_defs = ['enum', 'const', 'maxLength', 'minLength', 'pattern', 'format', 'minimum', 'maximum', 'exclusiveMinimum', 'exclusiveMaximum', 'multipleOf']

    def _jargon(self, term: Union[str, List[str]]) -> Union[str, List[str]]:
        if isinstance(term, list):
            return [self.jargon.get(t, t) for t in term]
        return self.jargon.get(term, term)

    def __call__(self, schema: Union[dict, List[dict]], prefix: str='', *, _path: Sequence[str]=()) -> str:
        if isinstance(schema, list):
            return self._handle_list(schema, prefix, _path)
        filtered = self._filter_unecessary(schema, _path)
        simple = self._handle_simple_dict(filtered, _path)
        if simple:
            return f'{prefix}{simple}'
        child_prefix = self._child_prefix(prefix, '  ')
        item_prefix = self._child_prefix(prefix, '- ')
        indent = len(prefix) * ' '
        with io.StringIO() as buffer:
            for i, (key, value) in enumerate(filtered.items()):
                child_path = [*_path, key]
                line_prefix = prefix if i == 0 else indent
                buffer.write(f'{line_prefix}{self._label(child_path)}:')
                if isinstance(value, dict):
                    filtered = self._filter_unecessary(value, child_path)
                    simple = self._handle_simple_dict(filtered, child_path)
                    buffer.write(f' {simple}' if simple else f'\n{self(value, child_prefix, _path=child_path)}')
                elif isinstance(value, list) and (key != 'type' or self._is_property(child_path)):
                    children = self._handle_list(value, item_prefix, child_path)
                    sep = ' ' if children.startswith('[') else '\n'
                    buffer.write(f'{sep}{children}')
                else:
                    buffer.write(f' {self._value(value, child_path)}\n')
            return buffer.getvalue()

    def _is_unecessary(self, path: Sequence[str]) -> bool:
        if self._is_property(path) or not path:
            return False
        key = path[-1]
        return any((key.startswith(k) for k in '$_')) or key in self._IGNORE

    def _filter_unecessary(self, schema: dict, path: Sequence[str]):
        return {key: value for key, value in schema.items() if not self._is_unecessary([*path, key])}

    def _handle_simple_dict(self, value: dict, path: Sequence[str]) -> Optional[str]:
        inline = any((p in value for p in self._guess_inline_defs))
        simple = not any((isinstance(v, (list, dict)) for v in value.values()))
        if inline or simple:
            return f'{{{', '.join(self._inline_attrs(value, path))}}}\n'
        return None

    def _handle_list(self, schemas: list, prefix: str='', path: Sequence[str]=()) -> str:
        if self._is_unecessary(path):
            return ''
        repr_ = repr(schemas)
        if all((not isinstance(e, (dict, list)) for e in schemas)) and len(repr_) < 60:
            return f'{repr_}\n'
        item_prefix = self._child_prefix(prefix, '- ')
        return ''.join((self(v, item_prefix, _path=[*path, f'[{i}]']) for i, v in enumerate(schemas)))

    def _is_property(self, path: Sequence[str]):
        """Check if the given path can correspond to an arbitrarily named property"""
        counter = 0
        for key in path[-2::-1]:
            if key not in {'properties', 'patternProperties'}:
                break
            counter += 1
        return counter % 2 == 1

    def _label(self, path: Sequence[str]) -> str:
        *parents, key = path
        if not self._is_property(path):
            norm_key = _separate_terms(key)
            return self._terms.get(key) or ' '.join(self._jargon(norm_key))
        if parents[-1] == 'patternProperties':
            return f'(regex {key!r})'
        return repr(key)

    def _value(self, value: Any, path: Sequence[str]) -> str:
        if path[-1] == 'type' and (not self._is_property(path)):
            type_ = self._jargon(value)
            return f'[{', '.join(type_)}]' if isinstance(value, list) else cast(str, type_)
        return repr(value)

    def _inline_attrs(self, schema: dict, path: Sequence[str]) -> Iterator[str]:
        for key, value in schema.items():
            child_path = [*path, key]
            yield f'{self._label(child_path)}: {self._value(value, child_path)}'

    def _child_prefix(self, parent_prefix: str, child_prefix: str) -> str:
        return len(parent_prefix) * ' ' + child_prefix