from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
class TriggerBuild(object):

    def __init__(self, request, module):
        self.module = module
        if request:
            self.request = request
        else:
            self.request = {}

    def to_request(self):
        return remove_nones_from_dict({u'source': TriggerSource(self.request.get('source', {}), self.module).to_request(), u'tags': self.request.get('tags'), u'images': self.request.get('images'), u'substitutions': self.request.get('substitutions'), u'queueTtl': self.request.get('queue_ttl'), u'logsBucket': self.request.get('logs_bucket'), u'timeout': self.request.get('timeout'), u'secrets': TriggerSecretsArray(self.request.get('secrets', []), self.module).to_request(), u'steps': TriggerStepsArray(self.request.get('steps', []), self.module).to_request(), u'artifacts': TriggerArtifacts(self.request.get('artifacts', {}), self.module).to_request(), u'options': TriggerOptions(self.request.get('options', {}), self.module).to_request()})

    def from_response(self):
        return remove_nones_from_dict({u'source': TriggerSource(self.request.get(u'source', {}), self.module).from_response(), u'tags': self.request.get(u'tags'), u'images': self.request.get(u'images'), u'substitutions': self.request.get(u'substitutions'), u'queueTtl': self.request.get(u'queueTtl'), u'logsBucket': self.request.get(u'logsBucket'), u'timeout': self.request.get(u'timeout'), u'secrets': TriggerSecretsArray(self.request.get(u'secrets', []), self.module).from_response(), u'steps': TriggerStepsArray(self.request.get(u'steps', []), self.module).from_response(), u'artifacts': TriggerArtifacts(self.request.get(u'artifacts', {}), self.module).from_response(), u'options': TriggerOptions(self.request.get(u'options', {}), self.module).from_response()})