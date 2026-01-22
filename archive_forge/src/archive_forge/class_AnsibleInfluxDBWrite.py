from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.general.plugins.module_utils.influxdb import InfluxDb
class AnsibleInfluxDBWrite(InfluxDb):

    def write_data_point(self, data_points):
        client = self.connect_to_influxdb()
        try:
            client.write_points(data_points)
        except Exception as e:
            self.module.fail_json(msg=to_native(e))