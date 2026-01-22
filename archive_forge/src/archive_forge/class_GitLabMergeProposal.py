import json
import os
import re
import time
from datetime import datetime
from typing import Optional
from ... import bedding
from ... import branch as _mod_branch
from ... import controldir, errors, urlutils
from ...forge import (Forge, ForgeLoginRequired, MergeProposal,
from ...git.urls import git_url_to_bzr_url
from ...trace import mutter
from ...transport import get_transport
class GitLabMergeProposal(MergeProposal):
    supports_auto_merge = True

    def __init__(self, gl, mr):
        self.gl = gl
        self._mr = mr

    def _update(self, **kwargs):
        try:
            self.gl._update_merge_request(self._mr['project_id'], self._mr['iid'], kwargs)
        except GitLabConflict as e:
            self.gl._handle_merge_request_conflict(e.reason, self.get_source_branch_url(), self._mr['target_project_id'])

    def __repr__(self):
        return '<{} at {!r}>'.format(type(self).__name__, self._mr['web_url'])

    @property
    def url(self):
        return self._mr['web_url']

    def get_web_url(self):
        return self._mr['web_url']

    def get_description(self):
        return self._mr['description']

    def set_description(self, description):
        try:
            self._update(description=description)
        except errors.UnexpectedHttpStatus as e:
            if e.code != 500:
                raise
            self._mr = self.gl._get_merge_request(self._mr['project_id'], self._mr['iid'])
            if self._mr['description'] != description:
                raise

    def get_commit_message(self):
        return self._mr.get('merge_commit_message')

    def set_commit_message(self, message):
        raise errors.UnsupportedOperation(self.set_commit_message, self)

    def get_title(self):
        return self._mr.get('title')

    def set_title(self, title):
        self._update(title=title)

    def _branch_url_from_project(self, project_id, branch_name, *, preferred_schemes=None):
        if project_id is None:
            return None
        project = self.gl._get_project(project_id)
        if preferred_schemes is None:
            preferred_schemes = DEFAULT_PREFERRED_SCHEMES
        for scheme in preferred_schemes:
            if scheme in SCHEME_MAP:
                return gitlab_url_to_bzr_url(project[SCHEME_MAP[scheme]], branch_name)
        raise KeyError

    def get_source_branch_url(self, *, preferred_schemes=None):
        return self._branch_url_from_project(self._mr['source_project_id'], self._mr['source_branch'], preferred_schemes=preferred_schemes)

    def get_source_revision(self):
        from breezy.git.mapping import default_mapping
        sha = self._mr['sha']
        if sha is None:
            return None
        return default_mapping.revision_id_foreign_to_bzr(sha.encode('ascii'))

    def get_target_branch_url(self, *, preferred_schemes=None):
        return self._branch_url_from_project(self._mr['target_project_id'], self._mr['target_branch'], preferred_schemes=preferred_schemes)

    def set_target_branch_name(self, name):
        self._update(target_branch=name)

    def _get_project_name(self, project_id):
        source_project = self.gl._get_project(project_id)
        return source_project['path_with_namespace']

    def get_source_project(self):
        return self._get_project_name(self._mr['source_project_id'])

    def get_target_project(self):
        return self._get_project_name(self._mr['target_project_id'])

    def is_merged(self):
        return self._mr['state'] == 'merged'

    def is_closed(self):
        return self._mr['state'] == 'closed'

    def reopen(self):
        return self._update(state_event='reopen')

    def close(self):
        self._update(state_event='close')

    def merge(self, commit_message=None, auto=False):
        ret = self.gl._merge_mr(self._mr['project_id'], self._mr['iid'], kwargs={'merge_commit_message': commit_message, 'merge_when_pipeline_succeeds': auto})
        self._mr.update(ret)

    def can_be_merged(self):
        if self._mr['merge_status'] == 'cannot_be_merged':
            return False
        elif self._mr['merge_status'] == 'can_be_merged':
            return True
        elif self._mr['merge_status'] in ('unchecked', 'cannot_be_merged_recheck', 'checking'):
            return None
        else:
            raise ValueError(self._mr['merge_status'])

    def get_merged_by(self):
        user = self._mr.get('merge_user')
        if user is None:
            return None
        return user['username']

    def get_merged_at(self):
        merged_at = self._mr.get('merged_at')
        if merged_at is None:
            return None
        return parse_timestring(merged_at)

    def post_comment(self, body):
        kwargs = {'body': body}
        self.gl._post_merge_request_note(self._mr['project_id'], self._mr['iid'], kwargs)