import enum
import os
import re
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.container import kubeconfig
from googlecloudsdk.api_lib.run import container_resource
from googlecloudsdk.api_lib.run import global_methods
from googlecloudsdk.api_lib.run import k8s_object
from googlecloudsdk.api_lib.run import revision
from googlecloudsdk.api_lib.run import service
from googlecloudsdk.api_lib.run import traffic
from googlecloudsdk.api_lib.services import enable_api
from googlecloudsdk.api_lib.services import exceptions as services_exceptions
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as c_exceptions
from googlecloudsdk.command_lib.functions.v2.deploy import env_vars_util
from googlecloudsdk.command_lib.run import config_changes
from googlecloudsdk.command_lib.run import exceptions as serverless_exceptions
from googlecloudsdk.command_lib.run import platforms
from googlecloudsdk.command_lib.run import pretty_print
from googlecloudsdk.command_lib.run import resource_args
from googlecloudsdk.command_lib.run import secrets_mapping
from googlecloudsdk.command_lib.run import volumes
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.args import map_util
from googlecloudsdk.command_lib.util.args import repeated
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
def _GetScalingChanges(args):
    """Returns the list of changes for scaling for given args."""
    result = []
    if 'min_instances' in args and args.min_instances is not None:
        scale_value = args.min_instances
        if scale_value.restore_default or scale_value.instance_count == 0:
            result.append(config_changes.DeleteTemplateAnnotationChange(revision.MIN_SCALE_ANNOTATION))
        else:
            result.append(config_changes.SetTemplateAnnotationChange(revision.MIN_SCALE_ANNOTATION, str(scale_value.instance_count)))
    if 'max_instances' in args and args.max_instances is not None:
        scale_value = args.max_instances
        if scale_value.restore_default:
            result.append(config_changes.DeleteTemplateAnnotationChange(revision.MAX_SCALE_ANNOTATION))
        else:
            result.append(config_changes.SetTemplateAnnotationChange(revision.MAX_SCALE_ANNOTATION, str(scale_value.instance_count)))
    return result