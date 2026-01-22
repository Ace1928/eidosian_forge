from __future__ import annotations
import contextlib
import enum
import json
import logging
import uuid
from typing import (
import numpy as np
import sqlalchemy
from langchain_core._api import deprecated, warn_deprecated
from sqlalchemy import SQLColumnExpression, delete, func
from sqlalchemy.dialects.postgresql import JSON, JSONB, UUID
from sqlalchemy.orm import Session, relationship
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables.config import run_in_executor
from langchain_core.utils import get_from_dict_or_env
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import maximal_marginal_relevance
def _create_filter_clause(self, filters: Any) -> Any:
    """Convert LangChain IR filter representation to matching SQLAlchemy clauses.

        At the top level, we still don't know if we're working with a field
        or an operator for the keys. After we've determined that we can
        call the appropriate logic to handle filter creation.

        Args:
            filters: Dictionary of filters to apply to the query.

        Returns:
            SQLAlchemy clause to apply to the query.
        """
    if isinstance(filters, dict):
        if len(filters) == 1:
            key, value = list(filters.items())[0]
            if key.startswith('$'):
                if key.lower() not in ['$and', '$or']:
                    raise ValueError(f'Invalid filter condition. Expected $and or $or but got: {key}')
            else:
                return self._handle_field_filter(key, filters[key])
            if not isinstance(value, list):
                raise ValueError(f'Expected a list, but got {type(value)} for value: {value}')
            if key.lower() == '$and':
                and_ = [self._create_filter_clause(el) for el in value]
                if len(and_) > 1:
                    return sqlalchemy.and_(*and_)
                elif len(and_) == 1:
                    return and_[0]
                else:
                    raise ValueError('Invalid filter condition. Expected a dictionary but got an empty dictionary')
            elif key.lower() == '$or':
                or_ = [self._create_filter_clause(el) for el in value]
                if len(or_) > 1:
                    return sqlalchemy.or_(*or_)
                elif len(or_) == 1:
                    return or_[0]
                else:
                    raise ValueError('Invalid filter condition. Expected a dictionary but got an empty dictionary')
            else:
                raise ValueError(f'Invalid filter condition. Expected $and or $or but got: {key}')
        elif len(filters) > 1:
            for key in filters.keys():
                if key.startswith('$'):
                    raise ValueError(f'Invalid filter condition. Expected a field but got: {key}')
            and_ = [self._handle_field_filter(k, v) for k, v in filters.items()]
            if len(and_) > 1:
                return sqlalchemy.and_(*and_)
            elif len(and_) == 1:
                return and_[0]
            else:
                raise ValueError('Invalid filter condition. Expected a dictionary but got an empty dictionary')
        else:
            raise ValueError('Got an empty dictionary for filters.')
    else:
        raise ValueError(f'Invalid type: Expected a dictionary but got type: {type(filters)}')