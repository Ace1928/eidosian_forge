from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
class Organizations(base.Group):
    """Manage Apigee organizations."""
    detailed_help = {'DESCRIPTION': '  {description}\n\n  `{command}` contains commands for managing Apigee organizations, the\n  highest-level grouping of Apigee objects. All API proxies, environments, and\n  so forth each belong to one organization.\n\n  Apigee organizations are distinct from Cloud Platform organizations, being\n  more similar to Cloud Platform projects in scope and purpose.\n          ', 'EXAMPLES': '  To list all accessible organizations and their associated Cloud Platform projects, run:\n\n      $ {command} list\n\n  To get a JSON array of all organizations whose Cloud Platform project names\n  contain the word ``sandbox\'\', run:\n\n      $ {command} list --format=json --filter="project:(sandbox)"\n  '}