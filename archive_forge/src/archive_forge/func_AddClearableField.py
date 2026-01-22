from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
def AddClearableField(parser, arg_name, property_name, resource, full_help):
    """Add arguments corresponding to a field that can be cleared."""
    args = [base.Argument('--{}'.format(arg_name), help='Set the {} for the {}.'.format(property_name, resource)), base.Argument('--clear-{}'.format(arg_name), help='Clear the {} from the {}.'.format(property_name, resource), action='store_true')]
    group = parser.add_mutually_exclusive_group(help=full_help)
    for arg in args:
        arg.AddToParser(group)