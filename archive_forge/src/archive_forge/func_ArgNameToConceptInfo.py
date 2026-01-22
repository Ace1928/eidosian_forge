from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.calliope.concepts import util
from googlecloudsdk.core import exceptions
import six
def ArgNameToConceptInfo(self, arg_name):
    return self._arg_name_lookup.get(util.NormalizeFormat(arg_name))