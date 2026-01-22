import unittest
from prov.model import *
from prov.dot import prov_to_dot
from prov.serializers import Registry
from prov.tests.examples import primer_example, primer_example_alternate
def add_further_attributes0(record):
    record.add_attributes([(EX_NS['tag1'], 'hello'), (EX_NS['tag2'], 'bye'), (EX_NS['tag2'], Literal('hola', langtag='es')), (EX2_NS['tag3'], 'hi'), (EX_NS['tag'], 1), (EX_NS['tag'], Literal(1, datatype=XSD_SHORT)), (EX_NS['tag'], Literal(1, datatype=XSD_DOUBLE)), (EX_NS['tag'], 1.0), (EX_NS['tag'], True), (EX_NS['tag'], EX_NS.uri + 'southampton')])
    add_further_attributes_with_qnames(record)