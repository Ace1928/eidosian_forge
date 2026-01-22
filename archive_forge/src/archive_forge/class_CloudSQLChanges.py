from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import abc
import argparse
import collections
from collections.abc import Collection, Container, Iterable, Mapping, MutableMapping
import copy
import dataclasses
import itertools
import json
import types
from typing import Any, ClassVar
from googlecloudsdk.api_lib.run import container_resource
from googlecloudsdk.api_lib.run import job
from googlecloudsdk.api_lib.run import k8s_object
from googlecloudsdk.api_lib.run import revision
from googlecloudsdk.api_lib.run import service
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.run import exceptions
from googlecloudsdk.command_lib.run import name_generator
from googlecloudsdk.command_lib.run import platforms
from googlecloudsdk.command_lib.run import secrets_mapping
from googlecloudsdk.command_lib.run import volumes
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.args import repeated
import six
@dataclasses.dataclass(frozen=True)
class CloudSQLChanges(TemplateConfigChanger):
    """Represents the intent to update the Cloug SQL instances.

  Attributes:
    project: Project to use as the default project for Cloud SQL instances.
    region: Region to use as the default region for Cloud SQL instances
    args: Args to the command.
  """
    add_cloudsql_instances: list[str]
    remove_cloudsql_instances: list[str]
    set_cloudsql_instances: list[str]
    clear_cloudsql_instances: bool | None = None

    @classmethod
    def FromArgs(cls, project: str | None=None, region: str | None=None, *, args: argparse.Namespace):
        """Returns a CloudSQLChanges object from the given args.

    Args:
      project: Optional project. If absent project must be specified in each
        Cloud SQL instance.
      region: Optional region. If absent region must be specified in each Cloud
        SQL instance.
      args: Command line args to parse CloudSQL flags from.
    """

        def AugmentArgs(arg_name):
            val = getattr(args, arg_name, None)
            if val is None:
                return None
            return [Augment(i) for i in val]

        def Augment(instance_str):
            instance = instance_str.split(':')
            if len(instance) == 3:
                return ':'.join(instance)
            elif len(instance) == 1:
                if not project:
                    raise exceptions.CloudSQLError('To specify a Cloud SQL instance by plain name, you must specify a project.')
                if not region:
                    raise exceptions.CloudSQLError('To specify a Cloud SQL instance by plain name, you must be deploying to a managed Cloud Run region.')
                return ':'.join(itertools.chain([project, region], instance))
            else:
                raise exceptions.CloudSQLError('Malformed CloudSQL instance string: {}'.format(instance_str))
        return cls(add_cloudsql_instances=AugmentArgs('add_cloudsql_instances'), remove_cloudsql_instances=AugmentArgs('remove_cloudsql_instances'), set_cloudsql_instances=AugmentArgs('set_cloudsql_instances'), clear_cloudsql_instances=getattr(args, 'clear_cloudsql_instances', None))

    def Adjust(self, resource):

        def GetCurrentInstances():
            annotation_val = resource.template.annotations.get(container_resource.CLOUDSQL_ANNOTATION)
            if annotation_val:
                return annotation_val.split(',')
            return []
        instances = repeated.ParsePrimitiveArgs(self, 'cloudsql-instances', GetCurrentInstances)
        if instances is not None:
            resource.template.annotations[container_resource.CLOUDSQL_ANNOTATION] = ','.join(instances)
        return resource