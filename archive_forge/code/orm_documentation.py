from oslo_utils import timeutils
import sqlalchemy.orm
from oslo_db.sqlalchemy import update_match
Emit an UPDATE statement matching the given specimen.

        This is a method-version of
        oslo_db.sqlalchemy.update_match.update_on_match(); see that function
        for usage details.

        