import contextlib
import ftplib
import gzip
import os
import re
import shutil
import sys
from urllib.request import urlopen
from urllib.request import urlretrieve
from urllib.request import urlcleanup
def update_pdb(self, file_format=None, with_assemblies=False):
    """Update your local copy of the PDB files.

        I guess this is the 'most wanted' function from this module.
        It gets the weekly lists of new and modified pdb entries and
        automatically downloads the according PDB files.
        You can call this module as a weekly cron job.
        """
    assert os.path.isdir(self.local_pdb)
    assert os.path.isdir(self.obsolete_pdb)
    file_format = self._print_default_format_warning(file_format)
    new, modified, obsolete = self.get_recent_changes()
    for pdb_code in new + modified:
        try:
            self.retrieve_pdb_file(pdb_code, file_format=file_format)
            if with_assemblies:
                assemblies = self.get_all_assemblies(file_format)
                for a_pdb_code, assembly_num in assemblies:
                    if a_pdb_code == pdb_code:
                        pl.retrieve_assembly_file(pdb_code, assembly_num, file_format=file_format, overwrite=True)
        except Exception as err:
            print(f'error {pdb_code}: {err}\n')
    for pdb_code in obsolete:
        if self.flat_tree:
            old_file = os.path.join(self.local_pdb, f'pdb{pdb_code}.{file_format}')
            new_dir = self.obsolete_pdb
        else:
            old_file = os.path.join(self.local_pdb, pdb_code[1:3], f'pdb{pdb_code}.{file_format}')
            new_dir = os.path.join(self.obsolete_pdb, pdb_code[1:3])
        new_file = os.path.join(new_dir, f'pdb{pdb_code}.{file_format}')
        if os.path.isfile(old_file):
            if not os.path.isdir(new_dir):
                os.mkdir(new_dir)
            try:
                shutil.move(old_file, new_file)
            except Exception:
                print(f'Could not move {old_file} to obsolete folder')
        elif os.path.isfile(new_file):
            if self._verbose:
                print(f'Obsolete file {old_file} already moved')
        elif self._verbose:
            print(f'Obsolete file {old_file} is missing')