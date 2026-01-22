import fixtures
import os
import testresources
import testscenarios
from oslo_db import exception
from oslo_db.sqlalchemy import enginefacade
from oslo_db.sqlalchemy import provision
from oslo_db.sqlalchemy import utils
class OpportunisticDBTestMixin(object):
    """Test mixin that integrates the test suite with testresources.

    There are three goals to this system:

    1. Allow creation of "stub" test suites that will run all the tests    in a
       parent suite against a specific    kind of database (e.g. Mysql,
       Postgresql), where the entire suite will be skipped if that    target
       kind of database is not available to the suite.

    2. provide a test with a process-local, anonymously named schema within a
       target database, so that the test can run concurrently with other tests
       without conflicting data

    3. provide compatibility with the testresources.OptimisingTestSuite, which
       organizes TestCase instances ahead of time into groups that all
       make use of the same type of database, setting up and tearing down
       a database schema once for the scope of any number of tests within.
       This technique is essential when testing against a non-SQLite database
       because building of a schema is expensive, and also is most ideally
       accomplished using the applications schema migration which are
       even more vastly slow than a straight create_all().

    This mixin provides the .resources attribute required by testresources when
    using the OptimisingTestSuite.The .resources attribute then provides a
    collection of testresources.TestResourceManager objects, which are defined
    here in oslo_db.sqlalchemy.provision.   These objects know how to find
    available database backends, build up temporary databases, and invoke
    schema generation and teardown instructions.   The actual "build the schema
    objects" part of the equation, and optionally a "delete from all the
    tables" step, is provided by the implementing application itself.


    """
    SKIP_ON_UNAVAILABLE_DB = True
    FIXTURE = OpportunisticDbFixture
    _collected_resources = None
    _instantiated_fixtures = None

    @property
    def resources(self):
        """Provide a collection of TestResourceManager objects.

        The collection here is memoized, both at the level of the test
        case itself, as well as in the fixture object(s) which provide
        those resources.

        """
        if self._collected_resources is not None:
            return self._collected_resources
        fixtures = self._instantiate_fixtures()
        self._collected_resources = []
        for fixture in fixtures:
            self._collected_resources.extend(fixture._get_resources())
        return self._collected_resources

    def setUp(self):
        self._setup_fixtures()
        super(OpportunisticDBTestMixin, self).setUp()

    def _get_default_provisioned_db(self):
        return self._db_default

    def _instantiate_fixtures(self):
        if self._instantiated_fixtures:
            return self._instantiated_fixtures
        self._instantiated_fixtures = utils.to_list(self.generate_fixtures())
        return self._instantiated_fixtures

    def generate_fixtures(self):
        return self.FIXTURE(test=self)

    def _setup_fixtures(self):
        testresources.setUpResources(self, self.resources, testresources._get_result())
        self.addCleanup(testresources.tearDownResources, self, self.resources, testresources._get_result())
        fixtures = self._instantiate_fixtures()
        for fixture in fixtures:
            self.useFixture(fixture)
            if not fixture._has_db_resource():
                msg = fixture._get_db_resource_not_available_reason()
                if self.SKIP_ON_UNAVAILABLE_DB:
                    self.skipTest(msg)
                else:
                    self.fail(msg)