import unittest
from prov.model import *
from prov.dot import prov_to_dot
from prov.serializers import Registry
from prov.tests.examples import primer_example, primer_example_alternate
def add_further_attributes_with_qnames(record):
    record.add_attributes([(EX_NS['tag'], EX2_NS['newyork']), (EX_NS['tag'], EX_NS['london'])])