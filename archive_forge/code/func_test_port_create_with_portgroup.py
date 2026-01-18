from ironicclient.tests.functional.osc.v1 import base
def test_port_create_with_portgroup(self):
    """Create port with specific port group UUID.

        Test steps:
        1) Create node in setUp().
        2) Create a port group.
        3) Create a port with specified port group.
        4) Check port properties for portgroup_uuid.
        """
    api_version = ' --os-baremetal-api-version 1.24'
    port_group = self.port_group_create(self.node['uuid'], params=api_version)
    port = self.port_create(self.node['uuid'], params='--port-group {0} {1}'.format(port_group['uuid'], api_version))
    self.assertEqual(port_group['uuid'], port['portgroup_uuid'])