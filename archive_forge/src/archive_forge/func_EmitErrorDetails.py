from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from functools import wraps
from apitools.base.py import exceptions
from googlecloudsdk.api_lib.util import exceptions as api_exceptions
def EmitErrorDetails(func):
    """Decorates a function for better errors."""

    @wraps(func)
    def Wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except exceptions.HttpError as error:
            raise api_exceptions.HttpException(error, '{message}{details?\n{?}}')
    return Wrapper