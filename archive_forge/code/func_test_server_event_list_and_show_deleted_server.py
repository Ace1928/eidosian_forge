from openstackclient.tests.functional.compute.v2 import common
def test_server_event_list_and_show_deleted_server(self):
    cmd_output = self.server_create(cleanup=False)
    server_id = cmd_output['id']
    self.openstack('server delete --wait ' + server_id)
    cmd_output = self.openstack('--os-compute-api-version 2.21 server event list ' + server_id, parse_output=True)
    request_id = None
    for each_event in cmd_output:
        self.assertNotIn('Message', each_event)
        self.assertNotIn('Project ID', each_event)
        self.assertNotIn('User ID', each_event)
        if each_event.get('Action') == 'delete':
            self.assertEqual(server_id, each_event.get('Server ID'))
            request_id = each_event.get('Request ID')
            break
    self.assertIsNotNone(request_id)
    cmd_output = self.openstack('--os-compute-api-version 2.21 server event show ' + server_id + ' ' + request_id, parse_output=True)
    self.assertEqual(request_id, cmd_output.get('request_id'))
    self.assertEqual('delete', cmd_output.get('action'))
    self.assertIsNotNone(cmd_output.get('events'))
    self.assertIsInstance(cmd_output.get('events'), list)