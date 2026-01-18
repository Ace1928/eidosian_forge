from .. import config
from .. import fixtures
from ..assertions import eq_
from ..assertions import is_true
from ..config import requirements
from ..provision import normalize_sequence
from ..schema import Column
from ..schema import Table
from ... import inspect
from ... import Integer
from ... import MetaData
from ... import Sequence
from ... import String
from ... import testing
@requirements.sequences_optional
def test_optional_seq(self, connection):
    r = connection.execute(self.tables.seq_opt_pk.insert(), dict(data='some data'))
    eq_(r.inserted_primary_key, (1,))