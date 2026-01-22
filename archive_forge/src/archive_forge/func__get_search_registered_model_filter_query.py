import logging
import urllib
import sqlalchemy
from sqlalchemy.future import select
import mlflow.store.db.utils
from mlflow.entities.model_registry.model_version_stages import (
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
from mlflow.store.artifact.utils.models import _parse_model_uri
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.model_registry import (
from mlflow.store.model_registry.abstract_store import AbstractStore
from mlflow.store.model_registry.dbmodels.models import (
from mlflow.utils.search_utils import SearchModelUtils, SearchModelVersionUtils, SearchUtils
from mlflow.utils.time import get_current_time_millis
from mlflow.utils.uri import extract_db_type_from_uri
from mlflow.utils.validation import (
@classmethod
def _get_search_registered_model_filter_query(cls, parsed_filters, dialect):
    attribute_filters = []
    tag_filters = {}
    for f in parsed_filters:
        type_ = f['type']
        key = f['key']
        comparator = f['comparator']
        value = f['value']
        if type_ == 'attribute':
            if key != 'name':
                raise MlflowException(f'Invalid attribute name: {key}', error_code=INVALID_PARAMETER_VALUE)
            if comparator not in ('=', '!=', 'LIKE', 'ILIKE'):
                raise MlflowException(f'Invalid comparator for attribute: {comparator}', error_code=INVALID_PARAMETER_VALUE)
            attr = getattr(SqlRegisteredModel, key)
            attr_filter = SearchUtils.get_sql_comparison_func(comparator, dialect)(attr, value)
            attribute_filters.append(attr_filter)
        elif type_ == 'tag':
            if comparator not in ('=', '!=', 'LIKE', 'ILIKE'):
                raise MlflowException.invalid_parameter_value(f'Invalid comparator for tag: {comparator}')
            if key not in tag_filters:
                key_filter = SearchUtils.get_sql_comparison_func('=', dialect)(SqlRegisteredModelTag.key, key)
                tag_filters[key] = [key_filter]
            val_filter = SearchUtils.get_sql_comparison_func(comparator, dialect)(SqlRegisteredModelTag.value, value)
            tag_filters[key].append(val_filter)
        else:
            raise MlflowException(f'Invalid token type: {type_}', error_code=INVALID_PARAMETER_VALUE)
    rm_query = select(SqlRegisteredModel).filter(*attribute_filters)
    if tag_filters:
        sql_tag_filters = (sqlalchemy.and_(*x) for x in tag_filters.values())
        tag_filter_query = select(SqlRegisteredModelTag.name).filter(sqlalchemy.or_(*sql_tag_filters)).group_by(SqlRegisteredModelTag.name).having(sqlalchemy.func.count(sqlalchemy.literal(1)) == len(tag_filters)).subquery()
        return rm_query.join(tag_filter_query, SqlRegisteredModel.name == tag_filter_query.c.name)
    else:
        return rm_query