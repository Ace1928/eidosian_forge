from openstackclient.tests.functional.volume.v2 import common
def test_volume_service_set(self):
    cmd_output = self.openstack('volume service list', parse_output=True)
    service_1 = cmd_output[0]['Binary']
    host_1 = cmd_output[0]['Host']
    raw_output = self.openstack('volume service set --enable ' + host_1 + ' ' + service_1)
    self.assertOutput('', raw_output)
    cmd_output = self.openstack('volume service list --long', parse_output=True)
    self.assertEqual('enabled', cmd_output[0]['Status'])
    self.assertIsNone(cmd_output[0]['Disabled Reason'])
    disable_reason = 'disable_reason'
    raw_output = self.openstack('volume service set --disable ' + '--disable-reason ' + disable_reason + ' ' + host_1 + ' ' + service_1)
    self.assertOutput('', raw_output)
    cmd_output = self.openstack('volume service list --long', parse_output=True)
    self.assertEqual('disabled', cmd_output[0]['Status'])
    self.assertEqual(disable_reason, cmd_output[0]['Disabled Reason'])