from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.ml_engine import jobs
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.ml_engine import flags
from googlecloudsdk.command_lib.ml_engine import jobs_util
from googlecloudsdk.command_lib.util.args import labels_util
def _GetAndValidateKmsKey(args):
    """Parse CMEK resource arg, and check if the arg was partially specified."""
    if hasattr(args.CONCEPTS, 'kms_key'):
        kms_ref = args.CONCEPTS.kms_key.Parse()
        if kms_ref:
            return kms_ref.RelativeName()
        else:
            for keyword in ['kms-key', 'kms-keyring', 'kms-location', 'kms-project']:
                if getattr(args, keyword.replace('-', '_'), None):
                    raise exceptions.InvalidArgumentException('--kms-key', 'Encryption key not fully specified.')