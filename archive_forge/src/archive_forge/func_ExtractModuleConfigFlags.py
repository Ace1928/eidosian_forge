from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
def ExtractModuleConfigFlags(parser):
    parent_group = parser.add_group(mutex=True, help='Config value group in Security Command Center.')
    AddConfigArgument(parent_group)
    AddClearConfigArgument(parent_group)