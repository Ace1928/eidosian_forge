import fixtures
import os
import testresources
import testscenarios
from oslo_db import exception
from oslo_db.sqlalchemy import enginefacade
from oslo_db.sqlalchemy import provision
from oslo_db.sqlalchemy import utils
def _teardown_resources(self):
    for name, resource in self._get_resources():
        dep = self._dependency_resources.pop(name)
        resource.finishedWith(dep)