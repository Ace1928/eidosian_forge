from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import functools
import itertools
import sys
import textwrap
from googlecloudsdk.api_lib.ml_engine import jobs
from googlecloudsdk.api_lib.ml_engine import versions_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.iam import completers as iam_completers
from googlecloudsdk.command_lib.iam import iam_util as core_iam_util
from googlecloudsdk.command_lib.kms import resource_args as kms_resource_args
from googlecloudsdk.command_lib.ml_engine import constants
from googlecloudsdk.command_lib.ml_engine import models_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
def ParseAcceleratorFlag(accelerator):
    """Validates and returns a accelerator config message object."""
    types = [c for c in _ACCELERATOR_TYPE_MAPPER.choices]
    if accelerator is None:
        return None
    raw_type = accelerator.get('type', None)
    if raw_type not in types:
        raise ArgumentError('The type of the accelerator can only be one of the following: {}.\n'.format(', '.join(["'{}'".format(c) for c in types])))
    accelerator_count = accelerator.get('count', 1)
    if accelerator_count <= 0:
        raise ArgumentError('The count of the accelerator must be greater than 0.\n')
    accelerator_msg = versions_api.GetMessagesModule().GoogleCloudMlV1AcceleratorConfig
    accelerator_type = arg_utils.ChoiceToEnum(raw_type, accelerator_msg.TypeValueValuesEnum)
    return accelerator_msg(count=accelerator_count, type=accelerator_type)