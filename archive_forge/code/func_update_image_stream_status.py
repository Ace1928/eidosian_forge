from __future__ import (absolute_import, division, print_function)
from datetime import datetime, timezone, timedelta
import traceback
import copy
from ansible.module_utils._text import to_native
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.module_utils.six import iteritems
from ansible_collections.community.okd.plugins.module_utils.openshift_common import AnsibleOpenshiftModule
from ansible_collections.community.okd.plugins.module_utils.openshift_images_common import (
from ansible_collections.community.okd.plugins.module_utils.openshift_docker_image import (
def update_image_stream_status(self, definition):
    kind = definition['kind']
    api_version = definition['apiVersion']
    namespace = definition['metadata']['namespace']
    name = definition['metadata']['name']
    self.changed = True
    result = definition
    if not self.check_mode:
        try:
            result = self.request('PUT', '/apis/{api_version}/namespaces/{namespace}/imagestreams/{name}/status'.format(api_version=api_version, namespace=namespace, name=name), body=definition, content_type='application/json').to_dict()
        except DynamicApiError as exc:
            msg = 'Failed to patch object: kind={0} {1}/{2}'.format(kind, namespace, name)
            self.fail_json(msg=msg, status=exc.status, reason=exc.reason)
        except Exception as exc:
            msg = 'Failed to patch object kind={0} {1}/{2} due to: {3}'.format(kind, namespace, name, exc)
            self.fail_json(msg=msg, error=to_native(exc))
    return result