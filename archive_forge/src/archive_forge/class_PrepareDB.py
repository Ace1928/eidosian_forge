import os
from ase import Atoms
from ase.ga import get_raw_score
from ase.ga import set_parametrization, set_neighbor_list
import ase.db
class PrepareDB:
    """ Class used to initialize a database.

        This class is used once to setup the database and create
        working directories.

        Parameters:

        db_file_name: Database file to use

    """

    def __init__(self, db_file_name, simulation_cell=None, **kwargs):
        if os.path.exists(db_file_name):
            raise IOError('DB file {0} already exists'.format(os.path.abspath(db_file_name)))
        self.db_file_name = db_file_name
        if simulation_cell is None:
            simulation_cell = Atoms()
        self.c = ase.db.connect(self.db_file_name)
        data = dict(kwargs)
        self.c.write(simulation_cell, data=data, simulation_cell=True)

    def add_unrelaxed_candidate(self, candidate, **kwargs):
        """ Add an unrelaxed starting candidate. """
        gaid = self.c.write(candidate, origin='StartingCandidateUnrelaxed', relaxed=0, generation=0, extinct=0, **kwargs)
        self.c.update(gaid, gaid=gaid)
        candidate.info['confid'] = gaid

    def add_relaxed_candidate(self, candidate, **kwargs):
        """ Add a relaxed starting candidate. """
        test_raw_score(candidate)
        if 'data' in candidate.info:
            data = candidate.info['data']
        else:
            data = {}
        gaid = self.c.write(candidate, origin='StartingCandidateRelaxed', relaxed=1, generation=0, extinct=0, key_value_pairs=candidate.info['key_value_pairs'], data=data, **kwargs)
        self.c.update(gaid, gaid=gaid)
        candidate.info['confid'] = gaid