from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from typing import Dict, Union
from googlecloudsdk.calliope import parser_extensions
Maps specified flags to API field paths of mutable fields.

  args.GetSpecifiedArgNames() returns a list of specified flags.

  For example, `gcloud alpha container fleet create --display-name my-fleet
  --format 'yaml(displayName)'`
  args.GetSpecifiedArgNames() = ['--display-name', '--format']

  Args:
    args: All arguments passed from CLI.
    flag_to_update_mask_paths: Mapping for a specific resource, such as user
      cluster, or node pool.

  Returns:
    A string that contains yaml field paths to be used in the API update
    request.
  