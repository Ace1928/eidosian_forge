import operator
from django.utils.hashable import make_hashable
class FieldDoesNotExist(Exception):
    """The requested model field does not exist"""
    pass