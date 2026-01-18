from urllib import parse
import ddt
from cinderclient import api_versions
from cinderclient import exceptions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
from cinderclient.v3 import volume_snapshots
from cinderclient.v3 import volumes
def test_volume_manager_upload_to_image(self):
    expected = {'os-volume_upload_image': {'force': False, 'container_format': 'bare', 'disk_format': 'raw', 'image_name': 'name', 'visibility': 'public', 'protected': True}}
    api_version = api_versions.APIVersion('3.1')
    cs = fakes.FakeClient(api_version)
    manager = volumes.VolumeManager(cs)
    fake_volume = volumes.Volume(manager, {'id': 1234, 'name': 'sample-volume'}, loaded=True)
    fake_volume.upload_to_image(False, 'name', 'bare', 'raw', visibility='public', protected=True)
    cs.assert_called_anytime('POST', '/volumes/1234/action', body=expected)