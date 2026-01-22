import ast
import base64
import json
import math
import operator
import re
import shlex
import sqlparse
from packaging.version import Version
from sqlparse.sql import (
from sqlparse.tokens import Token as TokenType
from mlflow.entities import RunInfo
from mlflow.entities.model_registry.model_version_stages import STAGE_DELETED_INTERNAL
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.store.db.db_types import MSSQL, MYSQL, POSTGRES, SQLITE
from mlflow.utils.mlflow_tags import (
class SearchExperimentsUtils(SearchUtils):
    VALID_SEARCH_ATTRIBUTE_KEYS = {'name', 'creation_time', 'last_update_time'}
    VALID_ORDER_BY_ATTRIBUTE_KEYS = {'name', 'experiment_id', 'creation_time', 'last_update_time'}
    NUMERIC_ATTRIBUTES = {'creation_time', 'last_update_time'}

    @classmethod
    def _invalid_statement_token_search_experiments(cls, token):
        if isinstance(token, Comparison) or token.is_whitespace or token.match(ttype=TokenType.Keyword, values=['AND']):
            return False
        return True

    @classmethod
    def _process_statement(cls, statement):
        tokens = _join_in_comparison_tokens(statement.tokens)
        invalids = list(filter(cls._invalid_statement_token_search_experiments, tokens))
        if len(invalids) > 0:
            invalid_clauses = ', '.join(map(str, invalids))
            raise MlflowException.invalid_parameter_value(f'Invalid clause(s) in filter string: {invalid_clauses}')
        return [cls._get_comparison(t) for t in tokens if isinstance(t, Comparison)]

    @classmethod
    def _get_identifier(cls, identifier, valid_attributes):
        tokens = identifier.split('.', maxsplit=1)
        if len(tokens) == 1:
            key = tokens[0]
            identifier = cls._ATTRIBUTE_IDENTIFIER
        else:
            entity_type, key = tokens
            valid_entity_types = ('attribute', 'tag', 'tags')
            if entity_type not in valid_entity_types:
                raise MlflowException.invalid_parameter_value(f"Invalid entity type '{entity_type}'. Valid entity types are {valid_entity_types}")
            identifier = cls._valid_entity_type(entity_type)
        key = cls._trim_backticks(cls._strip_quotes(key))
        if identifier == cls._ATTRIBUTE_IDENTIFIER and key not in valid_attributes:
            raise MlflowException.invalid_parameter_value(f"Invalid attribute key '{key}' specified. Valid keys are '{valid_attributes}'")
        return {'type': identifier, 'key': key}

    @classmethod
    def _get_comparison(cls, comparison):
        stripped_comparison = [token for token in comparison.tokens if not token.is_whitespace]
        cls._validate_comparison(stripped_comparison)
        left, comparator, right = stripped_comparison
        comp = cls._get_identifier(left.value, cls.VALID_SEARCH_ATTRIBUTE_KEYS)
        comp['comparator'] = comparator.value
        comp['value'] = cls._get_value(comp.get('type'), comp.get('key'), right)
        return comp

    @classmethod
    def parse_order_by_for_search_experiments(cls, order_by):
        token_value, is_ascending = cls._parse_order_by_string(order_by)
        identifier = cls._get_identifier(token_value.strip(), cls.VALID_ORDER_BY_ATTRIBUTE_KEYS)
        return (identifier['type'], identifier['key'], is_ascending)

    @classmethod
    def is_attribute(cls, key_type, comparator):
        if key_type == cls._ATTRIBUTE_IDENTIFIER:
            if comparator not in cls.VALID_STRING_ATTRIBUTE_COMPARATORS:
                raise MlflowException(f"Invalid comparator '{comparator}' not one of '{cls.VALID_STRING_ATTRIBUTE_COMPARATORS}'")
            return True
        return False

    @classmethod
    def _does_experiment_match_clause(cls, experiment, sed):
        key_type = sed.get('type')
        key = sed.get('key')
        value = sed.get('value')
        comparator = sed.get('comparator').upper()
        if cls.is_string_attribute(key_type, key, comparator):
            lhs = getattr(experiment, key)
        elif cls.is_numeric_attribute(key_type, key, comparator):
            lhs = getattr(experiment, key)
            value = float(value)
        elif cls.is_tag(key_type, comparator):
            if key not in experiment.tags:
                return False
            lhs = experiment.tags.get(key, None)
            if lhs is None:
                return experiment
        else:
            raise MlflowException(f"Invalid search expression type '{key_type}'", error_code=INVALID_PARAMETER_VALUE)
        return SearchUtils.get_comparison_func(comparator)(lhs, value)

    @classmethod
    def filter(cls, experiments, filter_string):
        if not filter_string:
            return experiments
        parsed = cls.parse_search_filter(filter_string)

        def experiment_matches(experiment):
            return all((cls._does_experiment_match_clause(experiment, s) for s in parsed))
        return list(filter(experiment_matches, experiments))

    @classmethod
    def _get_sort_key(cls, order_by_list):
        order_by = []
        parsed_order_by = map(cls.parse_order_by_for_search_experiments, order_by_list)
        for type_, key, ascending in parsed_order_by:
            if type_ == 'attribute':
                order_by.append((key, ascending))
            else:
                raise MlflowException.invalid_parameter_value(f'Invalid order_by entity: {type_}')
        if not any((key == 'experiment_id' for key, _ in order_by)):
            order_by.append(('experiment_id', False))

        class _Sorter:

            def __init__(self, obj, ascending):
                self.obj = obj
                self.ascending = ascending

            def __eq__(self, other):
                return other.obj == self.obj

            def __lt__(self, other):
                if self.obj is None:
                    return False
                elif other.obj is None:
                    return True
                elif self.ascending:
                    return self.obj < other.obj
                else:
                    return other.obj < self.obj

        def _apply_sorter(experiment, key, ascending):
            attr = getattr(experiment, key)
            return _Sorter(attr, ascending)
        return lambda experiment: tuple((_apply_sorter(experiment, k, asc) for k, asc in order_by))

    @classmethod
    def sort(cls, experiments, order_by_list):
        return sorted(experiments, key=cls._get_sort_key(order_by_list))