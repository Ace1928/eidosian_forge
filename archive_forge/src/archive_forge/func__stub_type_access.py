from datetime import datetime
from urllib import parse as urlparse
from cinderclient import client as base_client
from cinderclient.tests.unit import fakes
import cinderclient.tests.unit.utils as utils
def _stub_type_access(**kwargs):
    access = {'volume_type_id': '11111111-1111-1111-1111-111111111111', 'project_id': '00000000-0000-0000-0000-000000000000'}
    access.update(kwargs)
    return access