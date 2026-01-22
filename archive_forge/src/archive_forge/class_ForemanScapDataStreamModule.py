from __future__ import absolute_import, division, print_function
import hashlib
import json
import os
import operator
import re
import time
import traceback
from contextlib import contextmanager
from collections import defaultdict
from functools import wraps
from ansible.module_utils.basic import AnsibleModule, missing_required_lib, env_fallback
from ansible.module_utils._text import to_bytes, to_native
from ansible.module_utils import six
class ForemanScapDataStreamModule(ForemanTaxonomicEntityAnsibleModule):

    def __init__(self, **kwargs):
        foreman_spec = dict(original_filename=dict(type='str'), scap_file=dict(type='path'))
        foreman_spec.update(kwargs.pop('foreman_spec', {}))
        super(ForemanScapDataStreamModule, self).__init__(foreman_spec=foreman_spec, **kwargs)

    def run(self, **kwargs):
        entity = self.lookup_entity('entity')
        if not self.desired_absent:
            if not entity and 'scap_file' not in self.foreman_params:
                self.fail_json(msg='Content of scap_file not provided. XML containing SCAP content is required.')
            if 'scap_file' in self.foreman_params and 'original_filename' not in self.foreman_params:
                self.foreman_params['original_filename'] = os.path.basename(self.foreman_params['scap_file'])
            if 'scap_file' in self.foreman_params:
                with open(self.foreman_params['scap_file']) as input_file:
                    self.foreman_params['scap_file'] = input_file.read()
            if entity and 'scap_file' in self.foreman_params:
                digest = hashlib.sha256(self.foreman_params['scap_file'].encode('utf-8')).hexdigest()
                digest_stripped = hashlib.sha256(self.foreman_params['scap_file'].strip().encode('utf-8')).hexdigest()
                if entity['digest'] in [digest, digest_stripped]:
                    self.foreman_params.pop('scap_file')
        return super(ForemanScapDataStreamModule, self).run(**kwargs)