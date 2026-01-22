import unittest
from prov.model import *
from prov.dot import prov_to_dot
from prov.serializers import Registry
from prov.tests.examples import primer_example, primer_example_alternate
def add_locations(record):
    record.add_attributes([('prov:Location', 'Southampton'), ('prov:Location', 1), ('prov:Location', 1.0), ('prov:Location', True), ('prov:Location', EX_NS['london']), ('prov:Location', datetime.datetime.now()), ('prov:Location', EX_NS.uri + 'london'), ('prov:Location', Literal(2002, datatype=XSD['gYear']))])