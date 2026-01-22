import fixtures
import os
import testresources
import testscenarios
from oslo_db import exception
from oslo_db.sqlalchemy import enginefacade
from oslo_db.sqlalchemy import provision
from oslo_db.sqlalchemy import utils
class MySQLOpportunisticFixture(OpportunisticDbFixture):
    DRIVER = 'mysql'