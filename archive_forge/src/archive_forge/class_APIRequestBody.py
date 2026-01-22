from __future__ import annotations
import logging
from enum import Enum
from typing import (
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.tools.openapi.utils.openapi_utils import HTTPVerb, OpenAPISpec
class APIRequestBody(BaseModel):
    """A model for a request body."""
    description: Optional[str] = Field(alias='description')
    'The description of the request body.'
    properties: List[APIRequestBodyProperty] = Field(alias='properties')
    media_type: str = Field(alias='media_type')
    'The media type of the request body.'

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

    @classmethod
    def from_request_body(cls, request_body: RequestBody, spec: OpenAPISpec) -> 'APIRequestBody':
        """Instantiate from an OpenAPI RequestBody."""
        properties = []
        for media_type, media_type_obj in request_body.content.items():
            if media_type not in _SUPPORTED_MEDIA_TYPES:
                continue
            api_request_body_properties = cls._process_supported_media_type(media_type_obj, spec)
            properties.extend(api_request_body_properties)
        return cls(description=request_body.description, properties=properties, media_type=media_type)