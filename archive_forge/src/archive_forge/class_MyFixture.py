import os
import testresources
import testscenarios
import unittest
from unittest import mock
from oslo_db.sqlalchemy import enginefacade
from oslo_db.sqlalchemy import provision
from oslo_db.sqlalchemy import test_fixtures
from oslo_db.tests import base as test_base
class MyFixture(test_fixtures.OpportunisticDbFixture):

    def get_enginefacade(self):
        return normal_mgr