from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
class Deployments(base.Group):
    """Manage deployments of Apigee API proxies in runtime environments."""
    detailed_help = {'DESCRIPTION': '\n          {description}\n\n          `{command}` contains commands for enumerating and checking the status\n          of deployments of proxies to runtime environments.\n          ', 'EXAMPLES': '\n          To list all deployments for the active Cloud Platform project, run:\n\n              $ {command} list\n\n          To list all deployments in a particular environment of a particular\n          Apigee organization, run:\n\n              $ {command} list --environment=ENVIRONMENT --organization=ORG_NAME\n\n          To get the status of a specific deployment as a JSON object, run:\n\n              $ {command} describe --api=API_NAME --environment=ENVIRONMENT --format=json\n      '}