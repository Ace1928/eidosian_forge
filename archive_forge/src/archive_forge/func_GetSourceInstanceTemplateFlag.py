from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.instances import flags as instance_flags
def GetSourceInstanceTemplateFlag(custom_name=None):
    """Gets the --source-instance-template flag."""
    help_text = '  The url of the instance template that will be used to populate the fields of\n  the reservation. Instance properties can not be defined in addition to source\n  instance template.\n  '
    return base.Argument(custom_name or '--source-instance-template', help=help_text)