import fixtures
import os
import testresources
import testscenarios
from oslo_db import exception
from oslo_db.sqlalchemy import enginefacade
from oslo_db.sqlalchemy import provision
from oslo_db.sqlalchemy import utils
class AdHocDbFixture(SimpleDbFixture):
    """"Fixture which creates and disposes a database engine per test.

    Also allows a specific URL to be passed, meaning the fixture can
    be hardcoded to a specific SQLite file.

    For a SQLite, this fixture will create the named database upon setup
    and tear it down upon teardown.   For other databases, the
    database is assumed to exist already and will remain after teardown.

    """

    def __init__(self, url=None):
        if url:
            self.url = utils.make_url(url)
            driver = self.url.get_backend_name()
        else:
            driver = None
            self.url = None
        BaseDbFixture.__init__(self, driver=driver, ident=provision._random_ident())
        self.url = url

    def _generate_database_resource(self, _enginefacade):
        return provision.DatabaseResource(self.driver, _enginefacade, ad_hoc_url=self.url, provision_new_database=False)

    def _cleanup(self):
        self._teardown_resources()