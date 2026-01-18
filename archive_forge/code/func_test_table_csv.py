from pathlib import Path
import pytest
from ase import Atoms
from ase.build import bulk, molecule
from ase.db import connect
def test_table_csv(cli, dbfile):
    txt = cli.ase('db', dbfile, '--csv')
    print(txt)
    tokens = txt.split(', ')
    check_tokens(tokens)