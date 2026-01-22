from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.core.util import files
def GetFilePatternFlag():
    return GetFlagOrPositional(name='pattern', positional=False, required=False, help='      Pattern to use for writing files. May contain the following formatting\n      verbs %n: metadata.name, %s: metadata.namespace, %k: kind\n      (default "%n_%k.yaml")\n      ')