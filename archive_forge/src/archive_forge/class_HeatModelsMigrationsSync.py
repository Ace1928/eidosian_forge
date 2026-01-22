from alembic import command as alembic_api
from alembic import script as alembic_script
import fixtures
from oslo_db.sqlalchemy import enginefacade
from oslo_db.sqlalchemy import test_fixtures
from oslo_db.sqlalchemy import test_migrations
from oslotest import base as test_base
import sqlalchemy
import testtools
from heat.db import migration
from heat.db import models
class HeatModelsMigrationsSync(test_migrations.ModelsMigrationsSync):

    def setUp(self):
        super().setUp()
        self.engine = enginefacade.writer.get_engine()
        self.sessionmaker = enginefacade.writer.get_sessionmaker()

    def get_metadata(self):
        return models.BASE.metadata

    def get_engine(self):
        return self.engine

    def db_sync(self, engine):
        migration.db_sync(engine=engine)

    def include_object(self, object_, name, type_, reflected, compare_to):
        return True