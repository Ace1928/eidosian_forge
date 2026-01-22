from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_exceptions
from googlecloudsdk.api_lib.cloudbuild.v2 import client_util
from googlecloudsdk.api_lib.cloudbuild.v2 import input_util
from googlecloudsdk.core import yaml
def _ResourcesTransform(workflow):
    """Transform resources message."""
    resources_map = {}
    types = ['topic', 'secretVersion']
    for resource in workflow.get('resources', []):
        if 'name' not in resource:
            raise cloudbuild_exceptions.InvalidYamlError('Name is required for resource.')
        if any((t in resource for t in types)):
            resources_map[resource.pop('name')] = resource
        elif 'repository' in resource:
            if resource['repository'].startswith('projects/'):
                resource['repo'] = resource.pop('repository')
            elif resource['repository'].startswith('https://'):
                resource['url'] = resource.pop('repository')
            else:
                raise cloudbuild_exceptions.InvalidYamlError('Malformed repo/url resource: {}'.format(resource['repository']))
            resources_map[resource.pop('name')] = resource
        else:
            raise cloudbuild_exceptions.InvalidYamlError('Unknown resource. Accepted types: {types}'.format(types=','.join(types + ['repository'])))
    if resources_map:
        workflow['resources'] = resources_map