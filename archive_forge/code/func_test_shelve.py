from breezy import commands, osutils, tests, trace, ui
from breezy.tests import script
def test_shelve(self):
    self.run_script('\n            $ brz shelve -m \'shelve bar\'\n            2>Shelve? ([y]es, [N]o, [f]inish, [q]uit): yes\n            <y\n            2>Selected changes:\n            2> M  file\n            2>Shelve 1 change(s)? ([y]es, [N]o, [f]inish, [q]uit): yes\n            <y\n            2>Changes shelved with id "1".\n            ', null_output_matches_anything=True)
    self.run_script('\n            $ brz shelve --list\n              1: shelve bar\n            ')