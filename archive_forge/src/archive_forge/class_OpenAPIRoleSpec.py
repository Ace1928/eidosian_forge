from __future__ import annotations
import re
import json
import copy
import contextlib
import operator
from abc import ABC
from pydantic import BaseModel
from typing import Any, Dict, List, Optional, Union, Callable, Tuple, TYPE_CHECKING
from lazyops.utils import logger
from lazyops.utils.lazy import lazy_import
from lazyops.libs.fastapi_utils.types.user_roles import UserRole
class OpenAPIRoleSpec(ABC):
    role: Optional[Union['UserRole', str]] = None
    included_paths: Optional[List[Union[str, Dict[str, str]]]] = []
    excluded_paths: Optional[List[Union[str, Dict[str, str]]]] = []
    included_tags: Optional[List[str]] = []
    excluded_tags: Optional[List[str]] = []
    included_schemas: Optional[List[str]] = []
    excluded_schemas: Optional[List[str]] = []
    openapi_schema: Optional[Dict[str, Any]] = None
    description_callable: Optional[Callable] = None
    extra_schemas: Optional[List[Union[BaseModel, Dict[str, Any], str]]] = None

    def __init__(self, role: Optional['UserRole']=None, included_paths: Optional[List[str]]=None, excluded_paths: Optional[List[str]]=None, included_tags: Optional[List[str]]=None, excluded_tags: Optional[List[str]]=None, included_schemas: Optional[List[str]]=None, excluded_schemas: Optional[List[str]]=None, openapi_schema: Optional[Dict[str, Any]]=None, description_callable: Optional[Callable]=None, extra_schemas: Optional[List[Union[BaseModel, Dict[str, Any], str]]]=None, extra_schema_prefix: Optional[str]=None, extra_schema_name_mapping: Optional[Dict[str, str]]=None, extra_schema_ref_template: Optional[str]='#/components/schemas/{model}', extra_schema_example_mapping: Optional[Dict[str, Dict[str, Any]]]=None, extra_schema_example_callable: Optional[Callable]=None, **kwargs):
        self.role = role or UserRole.ANON
        self.included_paths = included_paths or []
        self.excluded_paths = excluded_paths or []
        self.included_tags = included_tags or []
        self.excluded_tags = excluded_tags or []
        self.included_schemas = included_schemas or []
        self.excluded_schemas = excluded_schemas or []
        self.openapi_schema = openapi_schema
        self.description_callable = description_callable
        if extra_schemas is not None:
            self.extra_schemas = extra_schemas
        self.extra_schema_prefix = extra_schema_prefix
        self.extra_schema_name_mapping = extra_schema_name_mapping
        self.extra_schemas_populated = False
        self.extra_schemas_data: Dict[str, Dict[str, Any]] = None
        self.extra_schema_ref_template = extra_schema_ref_template
        self.extra_schema_example_mapping = extra_schema_example_mapping
        self.extra_schema_example_callable = extra_schema_example_callable
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def has_description_callable(self) -> bool:
        """
        Check if the role spec has a description callable
        """
        return self.description_callable is not None

    def populate_extra_schemas(self):
        """
        Populate the extra schemas
        """
        if self.extra_schemas_populated:
            return
        if not self.extra_schemas:
            return
        self.extra_schemas_data = {}
        for schema in self.extra_schemas:
            if isinstance(schema, str):
                try:
                    schema = lazy_import(schema)
                except Exception as e:
                    logger.warning(f'Invalid Extra Schema: {schema}, {e}')
                    continue
            if isinstance(schema, type(BaseModel)):
                schema_name = schema.__name__
                try:
                    schema = schema.model_json_schema(ref_template=self.extra_schema_ref_template)
                except Exception as e:
                    logger.warning(f'Invalid Extra Schema: {schema}, {e}')
                    continue
            elif isinstance(schema, dict):
                if 'title' not in schema:
                    logger.warning(f'Invalid Extra Schema. Does not contain `title` in schema: {schema}')
                    continue
                schema_name = schema['title']
            else:
                logger.warning(f'Invalid Extra Schema: {schema}')
                continue
            if self.extra_schema_name_mapping and schema_name in self.extra_schema_name_mapping:
                schema['title'] = self.extra_schema_name_mapping[schema_name]
            elif self.extra_schema_prefix:
                schema['title'] = f'{self.extra_schema_prefix}{schema_name}'
            if self.extra_schema_example_callable:
                if (schema_example := self.extra_schema_example_callable(schema=schema, schema_name=schema_name)):
                    schema['example'] = schema_example
            elif self.extra_schema_example_mapping and schema_name in self.extra_schema_example_mapping:
                schema['example'] = self.extra_schema_example_mapping[schema_name]
            self.extra_schemas_data[schema_name] = schema
        self.extra_schemas_populated = True