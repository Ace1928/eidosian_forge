import pytest
import os
from ase.db import connect
def update_keys_in_db(db):
    new_keys = {}
    for i in range(50):
        new_keys.update({f'mynewkey_{i}': 'test'})
    for row in db.select():
        db.update(row.id, **new_keys)