import pytest
from rpy2.robjects import packages
def test_sql(self):
    sql = dbplyr.sql('count(*)')
    assert 'sql' in sql.rclass