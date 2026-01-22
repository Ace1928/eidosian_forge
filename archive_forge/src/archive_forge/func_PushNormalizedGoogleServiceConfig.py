from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import re
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.endpoints import exceptions
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import retry
import six
def PushNormalizedGoogleServiceConfig(service_name, project, config_dict, config_id=None):
    """Pushes a given normalized Google service configuration.

  Args:
    service_name: name of the service
    project: the producer project Id
    config_dict: the parsed contents of the Google Service Config file.
    config_id: The id name for the config

  Returns:
    Result of the ServicesConfigsCreate request (a Service object)
  """
    messages = GetMessagesModule()
    client = GetClientInstance()
    service_config = encoding.DictToMessage(config_dict, messages.Service)
    service_config.producerProjectId = project
    service_config.id = config_id
    create_request = messages.ServicemanagementServicesConfigsCreateRequest(serviceName=service_name, service=service_config)
    return client.services_configs.Create(create_request)