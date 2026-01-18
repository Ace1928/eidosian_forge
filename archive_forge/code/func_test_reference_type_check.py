import unittest
from nltk.translate.meteor_score import meteor_score
def test_reference_type_check(self):
    str_reference = [' '.join(ref) for ref in self.reference]
    self.assertRaises(TypeError, meteor_score, str_reference, self.candidate)