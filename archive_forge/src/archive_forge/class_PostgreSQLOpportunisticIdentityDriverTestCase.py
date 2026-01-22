from oslo_db.sqlalchemy import enginefacade
from oslo_db.sqlalchemy import test_fixtures as db_fixtures
from oslotest import base as test_base
from keystone.common import sql
from keystone.identity.backends import sql as sql_backend
from keystone.tests.unit.identity.backends import test_base as id_test_base
from keystone.tests.unit.ksfixtures import database
class PostgreSQLOpportunisticIdentityDriverTestCase(TestIdentityDriver):
    FIXTURE = db_fixtures.PostgresqlOpportunisticFixture