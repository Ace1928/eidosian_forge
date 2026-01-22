from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import copy
import os
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import messages as messages_util
from googlecloudsdk.command_lib.artifacts import docker_util
from googlecloudsdk.command_lib.code import builders
from googlecloudsdk.command_lib.code import common
from googlecloudsdk.command_lib.code import dataobject
from googlecloudsdk.command_lib.code import yaml_helper
from googlecloudsdk.command_lib.run import exceptions
from googlecloudsdk.command_lib.run import flags as run_flags
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
def GenerateService(settings):
    """Generate a service configuration from a Cloud Settings configuration."""
    service = copy.deepcopy(settings.service)
    metadata = service.metadata or RUN_MESSAGES_MODULE.ObjectMeta()
    metadata.name = settings.service_name
    metadata.namespace = str(settings.project)
    service.metadata = metadata
    _BuildSpecTemplate(service)
    if settings.service_account:
        service.spec.template.spec.serviceAccountName = settings.service_account
    container = service.spec.template.spec.containers[0]
    container.image = settings.image
    _FillContainerRequirements(container, settings)
    return service