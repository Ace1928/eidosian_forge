from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
class HashDelimitedArgList(arg_parsers.ArgList):
    DEFAULT_DELIM_CHAR = '#'