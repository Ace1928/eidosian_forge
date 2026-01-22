from __future__ import (absolute_import, division, print_function)
from datetime import datetime
from ansible_collections.community.okd.plugins.module_utils.openshift_docker_image import (
from ansible.module_utils.six import iteritems
def analyze_refs_pod_creators(self, resources):
    keys = ('ReplicationController', 'DeploymentConfig', 'DaemonSet', 'Deployment', 'ReplicaSet', 'StatefulSet', 'Job', 'CronJob')
    for k, objects in iteritems(resources):
        if k not in keys:
            continue
        for obj in objects:
            if k == 'CronJob':
                spec = obj['spec']['jobTemplate']['spec']['template']['spec']
            else:
                spec = obj['spec']['template']['spec']
            referrer = {'kind': obj['kind'], 'namespace': obj['metadata']['namespace'], 'name': obj['metadata']['name']}
            err = self.analyze_refs_from_pod_spec(spec, referrer)
            if err:
                return err
    return None