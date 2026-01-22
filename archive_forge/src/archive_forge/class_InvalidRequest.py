from ._base import *
from .models import component_name, rename_if_scope_child_component
from fastapi import status
class InvalidRequest(BaseError):
    """The JSON sent is not a valid Request object"""
    CODE = status.HTTP_400_BAD_REQUEST
    MESSAGE = 'Invalid Request'
    error_model = ErrorModel