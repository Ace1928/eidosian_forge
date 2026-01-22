from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import ipaddress
import re
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import completers
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.command_lib.compute.instance_templates import service_proxy_aux_data
def ValidateSourceInstanceFlags(args):
    """Validates --source-instance flag."""
    if getattr(args, 'source_instance', False):
        if getattr(args, 'machine_type', False):
            raise exceptions.ConflictingArgumentsException('--source-instance', '--machine-type')
        if getattr(args, 'labels', False):
            raise exceptions.ConflictingArgumentsException('--source-instance', '--labels')
        if getattr(args, 'configure_disk', False):
            for disk in args.configure_disk:
                if 'device-name' not in disk:
                    raise exceptions.RequiredArgumentException('device-name', '`--configure-disk` requires `device-name` to be set')
                instantiate_from = disk.get('instantiate-from')
                custom_image = disk.get('custom-image')
                if custom_image and instantiate_from != 'custom-image':
                    raise exceptions.InvalidArgumentException('--configure-disk', "Value for `instantiate-from` must be 'custom-image' if the key `custom-image` is specified.")
                if instantiate_from == 'custom-image' and custom_image is None:
                    raise exceptions.InvalidArgumentException('--configure-disk', "Value for 'custom-image' must be specified if `instantiate-from` has value `custom-image`.")