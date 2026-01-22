from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from typing import Dict, Union
from googlecloudsdk.calliope import parser_extensions
def HasBinauthzConfig(args) -> bool:
    return args.IsKnownAndSpecified('binauthz_evaluation_mode') or args.IsKnownAndSpecified('binauthz_policy_bindings')