from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.instances import flags as instance_flags
def GetResourcePolicyFlag(custom_name=None):
    """Gets the --resource-policies flag."""
    help_text = '  The resource policies to include in this reservation. If you omit this flag,\n  no resource policies are added. You can specify any string as the key, and\n  specify the name of a resource policy as the value.\n  '
    return base.Argument(custom_name or '--resource-policies', metavar='KEY=VALUE', type=arg_parsers.ArgDict(), help=help_text)