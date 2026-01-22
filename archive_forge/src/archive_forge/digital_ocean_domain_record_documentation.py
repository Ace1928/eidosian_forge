from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.digitalocean.plugins.module_utils.digital_ocean import (
Collect exact and similar records

        It returns an exact record if there is any match along with the record_id.
        It also returns multiple records if there is no exact match
        