from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import re
from googlecloudsdk.api_lib.util import exceptions as exceptions_util
from googlecloudsdk.calliope import exceptions as c_exceptions
from googlecloudsdk.core import exceptions
import six
class InvalidRuntimeLanguage(exceptions.Error):

    def __init__(self, invalid_runtime):
        super(InvalidRuntimeLanguage, self).__init__(f'Runtime language [{invalid_runtime}] is not supported')