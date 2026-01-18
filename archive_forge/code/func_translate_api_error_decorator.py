from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.util import exceptions as api_exceptions
from googlecloudsdk.api_lib.util import resource
from googlecloudsdk.core import exceptions as core_exceptions
from six.moves import urllib
def translate_api_error_decorator(function):

    def wrapper(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except Exception as e:
            core_exceptions.reraise(translate_error(e, translation_list, format_str=format_str, status_code_getter=status_code_getter))
    return wrapper