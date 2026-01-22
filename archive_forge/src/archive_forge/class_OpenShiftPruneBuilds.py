from __future__ import (absolute_import, division, print_function)
from datetime import datetime, timezone, timedelta
import traceback
import time
from ansible.module_utils._text import to_native
from ansible_collections.community.okd.plugins.module_utils.openshift_common import AnsibleOpenshiftModule
class OpenShiftPruneBuilds(OpenShiftBuilds):

    def __init__(self, **kwargs):
        super(OpenShiftPruneBuilds, self).__init__(**kwargs)

    def execute_module(self):
        kind = 'Build'
        api_version = 'build.openshift.io/v1'
        resource = self.find_resource(kind=kind, api_version=api_version, fail=True)
        self.max_creation_timestamp = None
        keep_younger_than = self.params.get('keep_younger_than')
        if keep_younger_than:
            now = datetime.now(timezone.utc).replace(tzinfo=None)
            self.max_creation_timestamp = now - timedelta(minutes=keep_younger_than)

        def _prunable_build(build):
            return build['status']['phase'] in ('Complete', 'Failed', 'Error', 'Cancelled')

        def _orphan_build(build):
            if not _prunable_build(build):
                return False
            config = build['status'].get('config', None)
            if not config:
                return True
            build_config = self.get_build_config(config['name'], config['namespace'])
            return len(build_config) == 0

        def _younger_build(build):
            if not self.max_creation_timestamp:
                return False
            creation_timestamp = datetime.strptime(build['metadata']['creationTimestamp'], '%Y-%m-%dT%H:%M:%SZ')
            return creation_timestamp < self.max_creation_timestamp
        predicates = [_prunable_build]
        if self.params.get('orphans'):
            predicates.append(_orphan_build)
        if self.max_creation_timestamp:
            predicates.append(_younger_build)
        params = dict(kind=kind, api_version=api_version, namespace=self.params.get('namespace'))
        result = self.kubernetes_facts(**params)
        candidates = result['resources']
        for pred in predicates:
            candidates = list(filter(pred, candidates))
        if self.check_mode:
            changed = len(candidates) > 0
            self.exit_json(changed=changed, builds=candidates)
        changed = False
        for build in candidates:
            changed = True
            try:
                name = build['metadata']['name']
                namespace = build['metadata']['namespace']
                resource.delete(name=name, namespace=namespace, body={})
            except DynamicApiError as exc:
                msg = 'Failed to delete Build %s/%s due to: %s' % (namespace, name, exc.body)
                self.fail_json(msg=msg, status=exc.status, reason=exc.reason)
            except Exception as e:
                msg = 'Failed to delete Build %s/%s due to: %s' % (namespace, name, to_native(e))
                self.fail_json(msg=msg, error=to_native(e), exception=e)
        self.exit_json(changed=changed, builds=candidates)