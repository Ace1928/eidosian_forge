from __future__ import annotations
import logging
from enum import Enum
from typing import (
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.tools.openapi.utils.openapi_utils import HTTPVerb, OpenAPISpec
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