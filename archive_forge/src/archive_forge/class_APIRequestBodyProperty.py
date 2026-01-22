from __future__ import annotations
import logging
from enum import Enum
from typing import (
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.tools.openapi.utils.openapi_utils import HTTPVerb, OpenAPISpec
class APIRequestBodyProperty(APIPropertyBase):
    """A model for a request body property."""
    properties: List['APIRequestBodyProperty'] = Field(alias='properties')
    'The sub-properties of the property.'
    references_used: List[str] = Field(alias='references_used')
    'The references used by the property.'

    @classmethod
    def _process_object_schema(cls, schema: Schema, spec: OpenAPISpec, references_used: List[str]) -> Tuple[Union[str, List[str], None], List['APIRequestBodyProperty']]:
        from openapi_pydantic import Reference
        properties = []
        required_props = schema.required or []
        if schema.properties is None:
            raise ValueError(f'No properties found when processing object schema: {schema}')
        for prop_name, prop_schema in schema.properties.items():
            if isinstance(prop_schema, Reference):
                ref_name = prop_schema.ref.split('/')[-1]
                if ref_name not in references_used:
                    references_used.append(ref_name)
                    prop_schema = spec.get_referenced_schema(prop_schema)
                else:
                    continue
            properties.append(cls.from_schema(schema=prop_schema, name=prop_name, required=prop_name in required_props, spec=spec, references_used=references_used))
        return (schema.type, properties)

    @classmethod
    def _process_array_schema(cls, schema: Schema, name: str, spec: OpenAPISpec, references_used: List[str]) -> str:
        from openapi_pydantic import Reference, Schema
        items = schema.items
        if items is not None:
            if isinstance(items, Reference):
                ref_name = items.ref.split('/')[-1]
                if ref_name not in references_used:
                    references_used.append(ref_name)
                    items = spec.get_referenced_schema(items)
                else:
                    pass
                return f'Array<{ref_name}>'
            else:
                pass
            if isinstance(items, Schema):
                array_type = cls.from_schema(schema=items, name=f'{name}Item', required=True, spec=spec, references_used=references_used)
                return f'Array<{array_type.type}>'
        return 'array'

    @classmethod
    def from_schema(cls, schema: Schema, name: str, required: bool, spec: OpenAPISpec, references_used: Optional[List[str]]=None) -> 'APIRequestBodyProperty':
        """Recursively populate from an OpenAPI Schema."""
        if references_used is None:
            references_used = []
        schema_type = schema.type
        properties: List[APIRequestBodyProperty] = []
        if schema_type == 'object' and schema.properties:
            schema_type, properties = cls._process_object_schema(schema, spec, references_used)
        elif schema_type == 'array':
            schema_type = cls._process_array_schema(schema, name, spec, references_used)
        elif schema_type in PRIMITIVE_TYPES:
            pass
        elif schema_type is None:
            pass
        else:
            raise ValueError(f'Unsupported type: {schema_type}')
        return cls(name=name, required=required, type=schema_type, default=schema.default, description=schema.description, properties=properties, references_used=references_used)