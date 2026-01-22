import sqlalchemy
from sqlalchemy.sql import true
from keystone.catalog.backends import base
from keystone.common import driver_hints
from keystone.common import sql
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.i18n import _
def _check_parent_region(self, session, region_ref):
    """Raise a NotFound if the parent region does not exist.

        If the region_ref has a specified parent_region_id, check that
        the parent exists, otherwise, raise a NotFound.
        """
    parent_region_id = region_ref.get('parent_region_id')
    if parent_region_id is not None:
        self._get_region(session, parent_region_id)