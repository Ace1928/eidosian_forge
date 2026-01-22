from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.calliope import arg_parsers
class RequestModel(enum.Enum):
    PATH_STYLE = 'PATH_STYLE'
    VIRTUAL_HOSTED_STYLE = 'VIRTUAL_HOSTED_STYLE'