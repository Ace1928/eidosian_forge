from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
def AddEntityFilterFlags(parser):
    """Adds flags for entity filters to the given parser."""
    parser.add_argument('--kinds', metavar='KIND', type=arg_parsers.ArgList(), help="\n      A list specifying what kinds will be included in the operation. When\n      omitted, all Kinds are included. For example, to operate on only the\n      'Customer' and 'Order' Kinds:\n\n        $ {command} --kinds='Customer','Order'\n      ")
    parser.add_argument('--namespaces', metavar='NAMESPACE', type=arg_parsers.ArgList(), help="\n      A list specifying what namespaces will be included in the operation.\n      When omitted, all namespaces are included in the operation,\n      including the default namespace. To specify that *only* the default\n      namespace should be operated on, use the special symbol '(default)'.\n      For example, to operate on entities from both the 'customers' and default\n      namespaces:\n\n        $ {command} --namespaces='(default)','customers'\n      ")