from __future__ import absolute_import, division, print_function
import copy
import time
import traceback
from datetime import datetime
from ansible_collections.kubernetes.core.plugins.module_utils.ansiblemodule import (
from ansible_collections.kubernetes.core.plugins.module_utils.args_common import (
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.client import (
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.core import (
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.exceptions import (
from ansible.module_utils._text import to_native
def wait_for_pod_deletion(self, pods, wait_timeout, wait_sleep):
    start = datetime.now()

    def _elapsed_time():
        return (datetime.now() - start).seconds
    response = None
    pod = pods.pop()
    while (_elapsed_time() < wait_timeout or wait_timeout == 0) and pods:
        if not pod:
            pod = pods.pop()
        try:
            response = self._api_instance.read_namespaced_pod(namespace=pod[0], name=pod[1])
            if not response:
                pod = None
            time.sleep(wait_sleep)
        except ApiException as exc:
            if exc.reason != 'Not Found':
                self._module.fail_json(msg='Exception raised: {0}'.format(exc.reason))
            pod = None
        except Exception as e:
            self._module.fail_json(msg='Exception raised: {0}'.format(to_native(e)))
    if not pods:
        return None
    return 'timeout reached while pods were still running.'