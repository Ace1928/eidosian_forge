import fixtures
import os
import testresources
import testscenarios
from oslo_db import exception
from oslo_db.sqlalchemy import enginefacade
from oslo_db.sqlalchemy import provision
from oslo_db.sqlalchemy import utils
class GeneratesSchemaFromMigrations(GeneratesSchema):
    """Mixin defining a fixture as generating a schema using migrations.

    This is a "capability" mixin that works in conjunction with classes
    that include BaseDbFixture as a base.

    """
    _BUILD_WITH_MIGRATIONS = True

    def generate_schema_migrations(self, engine):
        """A hook which should generate the model schema using migrations.


        This hook is called within the scope of creating the database
        assuming BUILD_WITH_MIGRATIONS is True.

        """