from datetime import datetime
from urllib import parse as urlparse
from cinderclient import client as base_client
from cinderclient.tests.unit import fakes
import cinderclient.tests.unit.utils as utils
def _bookmark_href(base_uri, tenant_id, backup_id):
    return '%s/%s/backups/%s' % (base_uri, tenant_id, backup_id)