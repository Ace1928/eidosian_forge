from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule, env_fallback
from ansible_collections.community.digitalocean.plugins.module_utils.digital_ocean import (
Deletes a DigitalOcean Kubernetes cluster
        API reference: https://docs.digitalocean.com/reference/api/api-reference/#operation/delete_kubernetes_cluster
        