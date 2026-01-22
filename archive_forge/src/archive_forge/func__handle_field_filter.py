from __future__ import annotations
import enum
import logging
import os
from hashlib import md5
from typing import (
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.utils import get_from_dict_or_env
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import DistanceStrategy
def _handle_field_filter(field: str, value: Any, param_number: int=1) -> Tuple[str, Dict]:
    """Create a filter for a specific field.

    Args:
        field: name of field
        value: value to filter
            If provided as is then this will be an equality filter
            If provided as a dictionary then this will be a filter, the key
            will be the operator and the value will be the value to filter by
        param_number: sequence number of parameters used to map between param
           dict and Cypher snippet

    Returns a tuple of
        - Cypher filter snippet
        - Dictionary with parameters used in filter snippet
    """
    if not isinstance(field, str):
        raise ValueError(f'field should be a string but got: {type(field)} with value: {field}')
    if field.startswith('$'):
        raise ValueError(f'Invalid filter condition. Expected a field but got an operator: {field}')
    if not field.isidentifier():
        raise ValueError(f'Invalid field name: {field}. Expected a valid identifier.')
    if isinstance(value, dict):
        if len(value) != 1:
            raise ValueError(f'Invalid filter condition. Expected a value which is a dictionary with a single key that corresponds to an operator but got a dictionary with {len(value)} keys. The first few keys are: {list(value.keys())[:3]}')
        operator, filter_value = list(value.items())[0]
        if operator not in SUPPORTED_OPERATORS:
            raise ValueError(f'Invalid operator: {operator}. Expected one of {SUPPORTED_OPERATORS}')
    else:
        operator = '$eq'
        filter_value = value
    if operator in COMPARISONS_TO_NATIVE:
        native = COMPARISONS_TO_NATIVE[operator]
        query_snippet = f'n.`{field}` {native} $param_{param_number}'
        query_param = {f'param_{param_number}': filter_value}
        return (query_snippet, query_param)
    elif operator == '$between':
        low, high = filter_value
        query_snippet = f'$param_{param_number}_low <= n.`{field}` <= $param_{param_number}_high'
        query_param = {f'param_{param_number}_low': low, f'param_{param_number}_high': high}
        return (query_snippet, query_param)
    elif operator in {'$in', '$nin', '$like', '$ilike'}:
        if operator in {'$in', '$nin'}:
            for val in filter_value:
                if not isinstance(val, (str, int, float)):
                    raise NotImplementedError(f'Unsupported type: {type(val)} for value: {val}')
        if operator in {'$in'}:
            query_snippet = f'n.`{field}` IN $param_{param_number}'
            query_param = {f'param_{param_number}': filter_value}
            return (query_snippet, query_param)
        elif operator in {'$nin'}:
            query_snippet = f'n.`{field}` NOT IN $param_{param_number}'
            query_param = {f'param_{param_number}': filter_value}
            return (query_snippet, query_param)
        elif operator in {'$like'}:
            query_snippet = f'n.`{field}` CONTAINS $param_{param_number}'
            query_param = {f'param_{param_number}': filter_value.rstrip('%')}
            return (query_snippet, query_param)
        elif operator in {'$ilike'}:
            query_snippet = f'toLower(n.`{field}`) CONTAINS $param_{param_number}'
            query_param = {f'param_{param_number}': filter_value.rstrip('%')}
            return (query_snippet, query_param)
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()