from sqlalchemy import Column
from sqlalchemy import event
from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy import Table
from sqlalchemy.sql import text
from ...testing.fixtures import AlterColRoundTripFixture
from ...testing.fixtures import TestBase
def test_rename_column(self):
    self._run_alter_col({}, {'name': 'newname'})