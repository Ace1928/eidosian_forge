from unittest import mock
from unittest.mock import patch
import uuid
import glance_store
from oslo_config import cfg
from glance.common import exception
from glance.db.sqlalchemy import api as db_api
from glance import scrubber
from glance.tests import utils as test_utils
@mock.patch.object(db_api, 'image_get')
def test_store_delete_notfound_exception(self, mock_image_get):
    uri = 'file://some/path/%s' % uuid.uuid4()
    id = 'helloworldid'
    ex = glance_store.NotFound(message='random')
    scrub = scrubber.Scrubber(glance_store)
    with patch.object(glance_store, 'delete_from_backend') as _mock_delete:
        _mock_delete.side_effect = ex
        scrub._scrub_image(id, [(id, '-', uri)])