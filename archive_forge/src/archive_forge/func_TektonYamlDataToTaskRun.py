from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_exceptions
from googlecloudsdk.api_lib.cloudbuild.v2 import client_util
from googlecloudsdk.api_lib.cloudbuild.v2 import input_util
from googlecloudsdk.core import log
def TektonYamlDataToTaskRun(data):
    """Convert Tekton yaml file into TaskRun message."""
    _VersionCheck(data)
    metadata = _MetadataTransform(data)
    spec = data['spec']
    if 'taskSpec' in spec:
        _TaskSpecTransform(spec['taskSpec'])
        managed_sidecars = _MetadataToSidecar(metadata)
        if managed_sidecars:
            spec['taskSpec']['managedSidecars'] = managed_sidecars
    elif 'taskRef' in spec:
        input_util.RefTransform(spec['taskRef'])
    else:
        raise cloudbuild_exceptions.InvalidYamlError('TaskSpec or TaskRef is required.')
    _ServiceAccountTransformTaskSpec(spec)
    input_util.ParamDictTransform(spec.get('params', []))
    messages = client_util.GetMessagesModule()
    schema_message = encoding.DictToMessage(spec, messages.TaskRun)
    input_util.UnrecognizedFields(schema_message)
    return schema_message