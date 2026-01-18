from manilaclient.tests.functional.osc import base
def test_services_set(self):
    services = self.list_services()
    service = [service for service in services if service['Binary'] == 'manila-data']
    first_service = service[0]
    self.openstack(f'share service set {first_service['Host']} {first_service['Binary']} --disable --disable-reason test')
    result = self.listing_result('share service', 'list --status disabled')
    self.assertEqual(first_service['ID'], result[0]['ID'])
    self.assertEqual('disabled', result[0]['Status'])
    self.assertEqual('test', result[0]['Disabled Reason'])
    self.openstack(f'share service set {first_service['Host']} {first_service['Binary']} --enable')