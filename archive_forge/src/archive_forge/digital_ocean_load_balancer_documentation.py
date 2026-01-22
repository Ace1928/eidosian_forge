from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.digitalocean.plugins.module_utils.digital_ocean import (
Deletes a DigitalOcean Load Balancer
        API reference: https://docs.digitalocean.com/reference/api/api-reference/#operation/delete_load_balancer
        