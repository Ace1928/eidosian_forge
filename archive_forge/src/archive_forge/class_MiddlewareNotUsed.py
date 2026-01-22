import operator
from django.utils.hashable import make_hashable
class MiddlewareNotUsed(Exception):
    """This middleware is not used in this server configuration"""
    pass