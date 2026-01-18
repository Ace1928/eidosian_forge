from sqlalchemy import bindparam
from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import MetaData
from sqlalchemy import select
from sqlalchemy import String
from sqlalchemy import Table
from sqlalchemy import testing
from sqlalchemy import text
from sqlalchemy.testing import eq_
from sqlalchemy.testing import fixtures
note this test should succeed for all RETURNING backends
        as of 2.0.  In
        Idf28379f8705e403a3c6a937f6a798a042ef2540 we changed rowcount to use
        len(rows) when we have implicit returning

        