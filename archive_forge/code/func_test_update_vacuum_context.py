import pytest
import os
from ase.db import connect
def test_update_vacuum_context():
    with connect(db_name) as db:
        write_entries_to_db(db)
    with connect(db_name) as db:
        update_keys_in_db(db)
    with connect(db_name) as db:
        check_update_function(db)