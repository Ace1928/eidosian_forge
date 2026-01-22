from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import ipaddress
import re
from googlecloudsdk.api_lib.composer import util as api_util
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.composer import parsers
from googlecloudsdk.command_lib.composer import util as command_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import properties
import six
def AddAutoscalingUpdateFlagsToGroup(update_type_group, release_track):
    """Adds flags related to updating autoscaling.

  Args:
    update_type_group: argument group, the group to which flags should be added.
    release_track: gcloud version to add flags to.
  """
    if release_track == base.ReleaseTrack.GA:
        ENVIRONMENT_SIZE_GA.choice_arg.AddToParser(update_type_group)
    elif release_track == base.ReleaseTrack.BETA:
        ENVIRONMENT_SIZE_BETA.choice_arg.AddToParser(update_type_group)
    elif release_track == base.ReleaseTrack.ALPHA:
        ENVIRONMENT_SIZE_ALPHA.choice_arg.AddToParser(update_type_group)
    update_group = update_type_group.add_argument_group(AUTOSCALING_FLAG_GROUP_DESCRIPTION)
    SCHEDULER_CPU.AddToParser(update_group)
    WORKER_CPU.AddToParser(update_group)
    WEB_SERVER_CPU.AddToParser(update_group)
    SCHEDULER_MEMORY.AddToParser(update_group)
    WORKER_MEMORY.AddToParser(update_group)
    WEB_SERVER_MEMORY.AddToParser(update_group)
    SCHEDULER_STORAGE.AddToParser(update_group)
    WORKER_STORAGE.AddToParser(update_group)
    WEB_SERVER_STORAGE.AddToParser(update_group)
    MIN_WORKERS.AddToParser(update_group)
    MAX_WORKERS.AddToParser(update_group)
    triggerer_params_group = update_group.add_argument_group(TRIGGERER_PARAMETERS_FLAG_GROUP_DESCRIPTION, mutex=True)
    triggerer_enabled_group = triggerer_params_group.add_argument_group(TRIGGERER_ENABLED_GROUP_DESCRIPTION)
    TRIGGERER_CPU.AddToParser(triggerer_enabled_group)
    TRIGGERER_COUNT.AddToParser(triggerer_enabled_group)
    TRIGGERER_MEMORY.AddToParser(triggerer_enabled_group)
    ENABLE_TRIGGERER.AddToParser(triggerer_enabled_group)
    DISABLE_TRIGGERER.AddToParser(triggerer_params_group)
    if release_track != base.ReleaseTrack.GA:
        dag_processor_params_group = update_group.add_argument_group(DAG_PROCESSOR_PARAMETERS_FLAG_GROUP_DESCRIPTION)
        DAG_PROCESSOR_CPU.AddToParser(dag_processor_params_group)
        DAG_PROCESSOR_COUNT.AddToParser(dag_processor_params_group)
        DAG_PROCESSOR_MEMORY.AddToParser(dag_processor_params_group)
        DAG_PROCESSOR_STORAGE.AddToParser(dag_processor_params_group)
    NUM_SCHEDULERS.AddToParser(update_group)