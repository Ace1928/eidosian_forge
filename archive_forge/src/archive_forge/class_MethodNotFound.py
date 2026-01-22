from ._base import *
from .models import component_name, rename_if_scope_child_component
from fastapi import status
class MethodNotFound(BaseError):
    """The method does not exist / is not available"""
    CODE = status.HTTP_405_METHOD_NOT_ALLOWED
    MESSAGE = 'Method not found'