from enum import Enum
from typing import Any, Dict, List, Literal, Mapping, Optional, Union
import requests
from langchain_core.pydantic_v1 import BaseModel, SecretStr, root_validator
from langchain_core.retrievers import Document
class DALMFilter(BaseModel):
    """Filters available for a DALM retrieval and generation.

    Arguments:
        field_name: The field to filter on. Can be 'document' or 'name' to filter
            on your document's raw text or title. Any other field will be presumed
            to be a metadata field you included when uploading your context data
        filter_type: Currently 'fuzzy_search' and 'strict_search' are supported.
            'fuzzy_search' means a fuzzy search on the provided field is performed.
            The exact strict doesn't need to exist in the document
            for this to find a match.
            Very useful for scanning a document for some keyword terms.
            'strict_search' means that the exact string must appear
            in the provided field.
            This is NOT an exact eq filter. ie a document with content
            "the happy dog crossed the street" will match on a strict_search of
            "dog" but won't match on "the dog".
            Python equivalent of `return search_string in full_string`.
        value: The actual value to search for in the context data/metadata
    """
    field_name: str
    filter_type: DALMFilterType
    value: str
    _is_metadata: bool = False

    @root_validator()
    def set_meta(cls, values: Dict) -> Dict:
        """document and name are reserved arcee keys. Anything else is metadata"""
        values['_is_meta'] = values.get('field_name') not in ['document', 'name']
        return values