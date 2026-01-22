import unittest
from nltk.corpus import nombank
class NombankDemo(unittest.TestCase):

    def test_numbers(self):
        self.assertEqual(len(nombank.instances()), 114574)
        self.assertEqual(len(nombank.rolesets()), 5577)
        self.assertEqual(len(nombank.nouns()), 4704)

    def test_instance(self):
        self.assertEqual(nombank.instances()[0].roleset, 'perc-sign.01')

    def test_framefiles_fileids(self):
        self.assertEqual(len(nombank.fileids()), 4705)
        self.assertTrue(all((fileid.endswith('.xml') for fileid in nombank.fileids())))