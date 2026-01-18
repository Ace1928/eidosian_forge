import unittest
from prov.model import *
from prov.dot import prov_to_dot
from prov.serializers import Registry
from prov.tests.examples import primer_example, primer_example_alternate
def test_primer_alternate(self):
    g1 = primer_example()
    g2 = primer_example_alternate()
    self.assertEqual(g1, g2)