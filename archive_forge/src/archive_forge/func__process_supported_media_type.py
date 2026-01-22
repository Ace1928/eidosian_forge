from __future__ import annotations
import logging
from enum import Enum
from typing import (
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.tools.openapi.utils.openapi_utils import HTTPVerb, OpenAPISpec
@classmethod
def _process_supported_media_type(cls, media_type_obj: MediaType, spec: OpenAPISpec) -> List[APIRequestBodyProperty]:
    """Process the media type of the request body."""
    from openapi_pydantic import Reference
    references_used = []
    schema = media_type_obj.media_type_schema
    if isinstance(schema, Reference):
        references_used.append(schema.ref.split('/')[-1])
        schema = spec.get_referenced_schema(schema)
    if schema is None:
        raise ValueError(f'Could not resolve schema for media type: {media_type_obj}')
    api_request_body_properties = []
    required_properties = schema.required or []
    if schema.type == 'object' and schema.properties:
        for prop_name, prop_schema in schema.properties.items():
            if isinstance(prop_schema, Reference):
                prop_schema = spec.get_referenced_schema(prop_schema)
            api_request_body_properties.append(APIRequestBodyProperty.from_schema(schema=prop_schema, name=prop_name, required=prop_name in required_properties, spec=spec))
    else:
        api_request_body_properties.append(APIRequestBodyProperty(name='body', required=True, type=schema.type, default=schema.default, description=schema.description, properties=[], references_used=references_used))
    return api_request_body_properties