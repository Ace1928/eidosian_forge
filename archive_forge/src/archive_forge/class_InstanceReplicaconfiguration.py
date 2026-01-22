from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import time
class InstanceReplicaconfiguration(object):

    def __init__(self, request, module):
        self.module = module
        if request:
            self.request = request
        else:
            self.request = {}

    def to_request(self):
        return remove_nones_from_dict({u'failoverTarget': self.request.get('failover_target'), u'mysqlReplicaConfiguration': InstanceMysqlreplicaconfiguration(self.request.get('mysql_replica_configuration', {}), self.module).to_request(), u'replicaNames': self.request.get('replica_names'), u'serviceAccountEmailAddress': self.request.get('service_account_email_address')})

    def from_response(self):
        return remove_nones_from_dict({u'failoverTarget': self.request.get(u'failoverTarget'), u'mysqlReplicaConfiguration': InstanceMysqlreplicaconfiguration(self.request.get(u'mysqlReplicaConfiguration', {}), self.module).from_response(), u'replicaNames': self.request.get(u'replicaNames'), u'serviceAccountEmailAddress': self.request.get(u'serviceAccountEmailAddress')})