from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddDescriptionFlagToParser(parser, resource_name):
    """Adds description flag."""
    base.Argument('--description', help='Text description of a {}.'.format(resource_name), category=base.COMMONLY_USED_FLAGS).AddToParser(parser)