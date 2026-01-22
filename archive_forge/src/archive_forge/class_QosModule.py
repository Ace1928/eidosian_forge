from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
class QosModule(BaseModule):

    def _get_qos_type(self, type):
        if type == 'storage':
            return otypes.QosType.STORAGE
        elif type == 'network':
            return otypes.QosType.NETWORK
        elif type == 'hostnetwork':
            return otypes.QosType.HOSTNETWORK
        elif type == 'cpu':
            return otypes.QosType.CPU
        return None

    def build_entity(self):
        """
        Abstract method from BaseModule called from create() and remove()

        Builds the QoS from the given params

        :return: otypes.QoS
        """
        return otypes.Qos(name=self.param('name'), id=self.param('id'), type=self._get_qos_type(self.param('type')), description=self.param('description'), max_iops=self.param('max_iops'), max_read_iops=self.param('read_iops'), max_read_throughput=self.param('read_throughput'), max_throughput=self.param('max_throughput'), max_write_iops=self.param('write_iops'), cpu_limit=self.param('cpu_limit'), inbound_average=self.param('inbound_average'), inbound_peak=self.param('inbound_peak'), inbound_burst=self.param('inbound_burst'), outbound_average=self.param('outbound_average'), outbound_peak=self.param('outbound_peak'), outbound_burst=self.param('outbound_burst'), outbound_average_linkshare=self.param('outbound_average_linkshare'), outbound_average_upperlimit=self.param('outbound_average_upperlimit'), outbound_average_realtime=self.param('outbound_average_realtime'))