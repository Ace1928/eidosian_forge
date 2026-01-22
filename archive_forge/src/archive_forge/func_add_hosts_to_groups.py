from __future__ import absolute_import, division, print_function
from ansible.plugins.inventory import BaseInventoryPlugin, Constructable
from ansible.module_utils.urls import open_url
import json
def add_hosts_to_groups(self):
    hostgroups = self.set_hostgroups()
    for hostgroup in hostgroups:
        members = self.call_url(self.url, self.url_username, self.url_password, url_path='/monitoring/list/hosts' + '?hostgroup_name=' + hostgroup)
        for member in members:
            self.inventory.add_host(member['host_name'], group=hostgroup)