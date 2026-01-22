from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_exceptions
from googlecloudsdk.api_lib.cloudbuild.v2 import client_util
from googlecloudsdk.api_lib.cloudbuild.v2 import input_util
from googlecloudsdk.core import log
def _MetadataTransform(data):
    """Helper funtion to transform the metadata."""
    spec = data['spec']
    if not spec:
        raise cloudbuild_exceptions.InvalidYamlError('spec is empty.')
    metadata = data.pop('metadata')
    if not metadata:
        raise cloudbuild_exceptions.InvalidYamlError('Metadata is missing in yaml.')
    annotations = metadata.get('annotations', {})
    if _WORKER_POOL_ANNOTATION in annotations:
        spec['workerPool'] = annotations[_WORKER_POOL_ANNOTATION]
    spec['annotations'] = annotations
    if _MACHINE_TYPE in annotations:
        spec['worker'] = {'machineType': annotations[_MACHINE_TYPE]}
    security = {}
    if _PRIVILEGE_MODE in annotations:
        security['privilegeMode'] = annotations[_PRIVILEGE_MODE].upper()
    if security:
        spec['security'] = security
    provenance = {}
    if _PROVENANCE_ENABLED in annotations:
        provenance['enabled'] = annotations[_PROVENANCE_ENABLED].upper()
    if _PROVENANCE_STORAGE in annotations:
        provenance['storage'] = annotations[_PROVENANCE_STORAGE].upper()
    if _PROVENANCE_REGION in annotations:
        provenance['region'] = annotations[_PROVENANCE_REGION].upper()
    if provenance:
        spec['provenance'] = provenance
    return metadata