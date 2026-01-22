import email.message
import logging
import pathlib
import traceback
import urllib.parse
import warnings
from typing import Any, Callable, Dict, Iterator, Literal, Optional, Tuple, Type, Union
import requests
from gitlab import types
def _transform_types(data: Dict[str, Any], custom_types: Dict[str, Any], *, transform_data: bool, transform_files: Optional[bool]=True) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Copy the data dict with attributes that have custom types and transform them
    before being sent to the server.

    ``transform_files``: If ``True`` (default), also populates the ``files`` dict for
    FileAttribute types with tuples to prepare fields for requests' MultipartEncoder:
    https://toolbelt.readthedocs.io/en/latest/user.html#multipart-form-data-encoder

    ``transform_data``: If ``True`` transforms the ``data`` dict with fields
    suitable for encoding as query parameters for GitLab's API:
    https://docs.gitlab.com/ee/api/#encoding-api-parameters-of-array-and-hash-types

    Returns:
        A tuple of the transformed data dict and files dict"""
    data = data.copy()
    if not transform_files and (not transform_data):
        return (data, {})
    files = {}
    for attr_name, attr_class in custom_types.items():
        if attr_name not in data:
            continue
        gitlab_attribute = attr_class(data[attr_name])
        if isinstance(gitlab_attribute, types.FileAttribute) and transform_files:
            key = gitlab_attribute.get_file_name(attr_name)
            files[attr_name] = (key, data.pop(attr_name))
            continue
        if not transform_data:
            continue
        if isinstance(gitlab_attribute, types.GitlabAttribute):
            key, value = gitlab_attribute.get_for_api(key=attr_name)
            if key != attr_name:
                del data[attr_name]
            data[key] = value
    return (data, files)