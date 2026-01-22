import fixtures
import os
import testresources
import testscenarios
from oslo_db import exception
from oslo_db.sqlalchemy import enginefacade
from oslo_db.sqlalchemy import provision
from oslo_db.sqlalchemy import utils
class BaseDbFixture(fixtures.Fixture):
    """Base database provisioning fixture.

    This serves as the base class for the other fixtures, but by itself
    does not implement _setUp(). It provides the basis for the flags
    implemented by the various capability mixins (GenerateSchema,
    DeletesFromSchema, etc.) as well as providing an abstraction over
    the provisioning objects, which are specific to testresources.
    Overall, consumers of this fixture just need to use the right classes
    and the testresources mechanics are taken care of.

    """
    DRIVER = 'sqlite'
    _DROP_SCHEMA_PER_TEST = True
    _BUILD_SCHEMA = False
    _BUILD_WITH_MIGRATIONS = False
    _database_resources = {}
    _db_not_available = {}
    _schema_resources = {}

    def __init__(self, driver=None, ident=None):
        super(BaseDbFixture, self).__init__()
        self.driver = driver or self.DRIVER
        self.ident = ident or 'default'
        self.resource_key = (self.driver, self.__class__, self.ident)

    def get_enginefacade(self):
        """Return an enginefacade._TransactionContextManager.

        This is typically a global variable like "context_manager" declared
        in the db/api.py module and is the object returned by
        enginefacade.transaction_context().

        If left not implemented, the global enginefacade manager is used.

        For the case where a project uses per-object or per-test enginefacades
        like Gnocchi, the get_per_test_enginefacade()
        method should also be implemented.


        """
        return enginefacade._context_manager

    def get_per_test_enginefacade(self):
        """Return an enginefacade._TransactionContextManager per test.

        This facade should be the one that the test expects the code to
        use.   Usually this is the same one returned by get_engineafacade()
        which is the default.  For special applications like Gnocchi,
        this can be overridden to provide an instance-level facade.

        """
        return self.get_enginefacade()

    def _get_db_resource_not_available_reason(self):
        return self._db_not_available.get(self.resource_key, None)

    def _has_db_resource(self):
        return self._database_resources.get(self.resource_key, None) is not None

    def _generate_schema_resource(self, database_resource):
        return provision.SchemaResource(database_resource, None if not self._BUILD_SCHEMA else self.generate_schema_create_all if not self._BUILD_WITH_MIGRATIONS else self.generate_schema_migrations, self._DROP_SCHEMA_PER_TEST)

    def _get_resources(self):
        key = self.resource_key
        if key not in self._database_resources:
            _enginefacade = self.get_enginefacade()
            try:
                self._database_resources[key] = self._generate_database_resource(_enginefacade)
            except exception.BackendNotAvailable as bne:
                self._database_resources[key] = None
                self._db_not_available[key] = str(bne)
        database_resource = self._database_resources[key]
        if database_resource is None:
            return []
        else:
            if key in self._schema_resources:
                schema_resource = self._schema_resources[key]
            else:
                schema_resource = self._schema_resources[key] = self._generate_schema_resource(database_resource)
            return [('_schema_%s' % self.ident, schema_resource), ('_db_%s' % self.ident, database_resource)]