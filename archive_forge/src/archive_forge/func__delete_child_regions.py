import sqlalchemy
from sqlalchemy.sql import true
from keystone.catalog.backends import base
from keystone.common import driver_hints
from keystone.common import sql
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.i18n import _
def _delete_child_regions(self, session, region_id, root_region_id):
    """Delete all child regions.

        Recursively delete any region that has the supplied region
        as its parent.
        """
    children = session.query(Region).filter_by(parent_region_id=region_id)
    for child in children:
        if child.id == root_region_id:
            return
        self._delete_child_regions(session, child.id, root_region_id)
        session.delete(child)