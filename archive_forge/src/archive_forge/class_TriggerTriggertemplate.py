from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
class TriggerTriggertemplate(object):

    def __init__(self, request, module):
        self.module = module
        if request:
            self.request = request
        else:
            self.request = {}

    def to_request(self):
        return remove_nones_from_dict({u'projectId': self.request.get('project_id'), u'repoName': self.request.get('repo_name'), u'dir': self.request.get('dir'), u'invertRegex': self.request.get('invert_regex'), u'branchName': self.request.get('branch_name'), u'tagName': self.request.get('tag_name'), u'commitSha': self.request.get('commit_sha')})

    def from_response(self):
        return remove_nones_from_dict({u'projectId': self.request.get(u'projectId'), u'repoName': self.request.get(u'repoName'), u'dir': self.request.get(u'dir'), u'invertRegex': self.request.get(u'invertRegex'), u'branchName': self.request.get(u'branchName'), u'tagName': self.request.get(u'tagName'), u'commitSha': self.request.get(u'commitSha')})