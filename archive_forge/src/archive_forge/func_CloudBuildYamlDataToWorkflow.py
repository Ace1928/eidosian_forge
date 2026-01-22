from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_exceptions
from googlecloudsdk.api_lib.cloudbuild.v2 import client_util
from googlecloudsdk.api_lib.cloudbuild.v2 import input_util
from googlecloudsdk.core import yaml
def CloudBuildYamlDataToWorkflow(workflow):
    """Convert cloudbuild.yaml file into Workflow message."""
    _WorkflowTransform(workflow)
    messages = client_util.GetMessagesModule()
    schema_message = encoding.DictToMessage(workflow, messages.Workflow)
    input_util.UnrecognizedFields(schema_message)
    return schema_message