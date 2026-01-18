import os
from breezy import branchbuilder, errors, log, osutils, tests
from breezy.tests import features, test_log
def test_stdout_encoding(self):
    brz = self.run_bzr
    osutils._cached_user_encoding = 'cp1251'
    brz('init')
    self.build_tree(['a'])
    brz('add a')
    brz(['commit', '-m', 'Тест'])
    stdout, stderr = self.run_bzr_raw('log', encoding='cp866')
    message = stdout.splitlines()[-1]
    test_in_cp866 = b'\x92\xa5\xe1\xe2'
    test_in_cp1251 = b'\xd2\xe5\xf1\xf2'
    self.assertEqual(test_in_cp866, message[2:])
    self.assertEqual(-1, stdout.find(test_in_cp1251))