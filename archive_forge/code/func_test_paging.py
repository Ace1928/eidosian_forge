import pytest
from ase import Atoms
from ase.db import connect
from ase.db.web import Session
def test_paging(database):
    """Test paging."""
    pytest.importorskip('flask')
    session = Session('name')
    project = {'default_columns': ['bar'], 'handle_query_function': handle_query}
    session.update('query', '', {'query': ''}, project)
    table = session.create_table(database, 'id', ['foo'])
    assert len(table.rows) == 2
    session.update('limit', '1', {}, project)
    session.update('page', '1', {}, project)
    table = session.create_table(database, 'id', ['foo'])
    assert len(table.rows) == 1
    session.update('query', '', {'query': 'id=1'}, project)
    table = session.create_table(database, 'id', ['foo'])
    assert len(table.rows) == 1