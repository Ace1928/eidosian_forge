from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import logging
import os
from apitools.gen import gen_client
from googlecloudsdk.api_lib.regen import api_def
from googlecloudsdk.api_lib.regen import resource_generator
from googlecloudsdk.core.util import files
from mako import runtime
from mako import template
import six
def _MakeGapicClientDef(root_package, api_name, api_version):
    """Makes a GapicClientDef."""
    gapic_root_package = '.'.join(root_package.split('.')[:-1])
    class_path = '.'.join([gapic_root_package, 'gapic_wrappers', api_name, api_version])
    return api_def.GapicClientDef(class_path=class_path)