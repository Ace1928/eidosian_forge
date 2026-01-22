import fixtures
import os
import testresources
import testscenarios
from oslo_db import exception
from oslo_db.sqlalchemy import enginefacade
from oslo_db.sqlalchemy import provision
from oslo_db.sqlalchemy import utils
class DeletesFromSchema(ResetsData):
    """Mixin defining a fixture that can delete from all tables in place.

    When DeletesFromSchema is present in a fixture,
    _DROP_SCHEMA_PER_TEST is now False; this means that the
    "teardown" flag of provision.SchemaResource will be False, which
    prevents SchemaResource from dropping all objects within the schema
    after each test.

    This is a "capability" mixin that works in conjunction with classes
    that include BaseDbFixture as a base.

    """

    def reset_schema_data(self, engine, facade):
        self.delete_from_schema(engine)

    def delete_from_schema(self, engine):
        """A hook which should delete all data from an existing schema.

        Should *not* drop any objects, just remove data from tables
        that needs to be reset between tests.
        """