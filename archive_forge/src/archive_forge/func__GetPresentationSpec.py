from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import sys
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.command_lib.meta import cache_util
from googlecloudsdk.command_lib.util import parameter_info_lib
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import module_util
from googlecloudsdk.core.console import console_io
import six
def _GetPresentationSpec(resource_spec_path, **kwargs):
    """Build a presentation spec."""
    resource_spec = module_util.ImportModule(resource_spec_path)
    if callable(resource_spec):
        resource_spec = resource_spec()
    flag_name_overrides = kwargs.pop('flag_name_overrides', '')
    flag_name_overrides = {o.split(':')[0]: o.split(':')[1] if ':' in o else '' for o in flag_name_overrides.split(';') if o}
    prefixes = kwargs.pop('prefixes', False)
    return presentation_specs.ResourcePresentationSpec(kwargs.pop('name', resource_spec.name), resource_spec, 'help text', flag_name_overrides=flag_name_overrides, prefixes=prefixes, **kwargs)