import os
from alembic import command as alembic_api
from alembic import script as alembic_script
import fixtures
from oslo_db.sqlalchemy import enginefacade
from oslo_db.sqlalchemy import test_fixtures
from oslo_log import log as logging
import sqlalchemy
from keystone.common import sql
from keystone.common.sql import upgrades
import keystone.conf
from keystone.tests import unit
import keystone.application_credential.backends.sql  # noqa: F401
import keystone.assignment.backends.sql  # noqa: F401
import keystone.assignment.role_backends.sql_model  # noqa: F401
import keystone.catalog.backends.sql  # noqa: F401
import keystone.credential.backends.sql  # noqa: F401
import keystone.endpoint_policy.backends.sql  # noqa: F401
import keystone.federation.backends.sql  # noqa: F401
import keystone.identity.backends.sql_model  # noqa: F401
import keystone.identity.mapping_backends.sql  # noqa: F401
import keystone.limit.backends.sql  # noqa: F401
import keystone.oauth1.backends.sql  # noqa: F401
import keystone.policy.backends.sql  # noqa: F401
import keystone.resource.backends.sql_model  # noqa: F401
import keystone.resource.config_backends.sql  # noqa: F401
import keystone.revoke.backends.sql  # noqa: F401
import keystone.trust.backends.sql  # noqa: F401
class BannedDBSchemaOperations(fixtures.Fixture):
    """Ban some operations for migrations."""

    def __init__(self, banned_ops, revision):
        super().__init__()
        self._banned_ops = banned_ops or {}
        self._revision = revision

    @staticmethod
    def _explode(op, revision):

        def fail(*a, **kw):
            msg = "Operation '%s' is not allowed in migration %s"
            raise DBOperationNotAllowed(msg % (op, revision))
        return fail

    def setUp(self):
        super().setUp()
        for op in self._banned_ops:
            self.useFixture(fixtures.MonkeyPatch('alembic.op.%s' % op, self._explode(op, self._revision)))