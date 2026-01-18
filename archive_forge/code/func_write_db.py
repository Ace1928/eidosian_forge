import ase.db
from ase.io.formats import string2index
def write_db(filename, images, append=False, **kwargs):
    con = ase.db.connect(filename, serial=True, append=append, **kwargs)
    for atoms in images:
        con.write(atoms)