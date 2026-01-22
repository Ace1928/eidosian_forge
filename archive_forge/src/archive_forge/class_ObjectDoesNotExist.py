import operator
from django.utils.hashable import make_hashable
class ObjectDoesNotExist(Exception):
    """The requested object does not exist"""
    silent_variable_failure = True