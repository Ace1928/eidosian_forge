from pyxnat.core.help import GraphData, PaintGraph, SchemasInspector
from pyxnat import Interface
from pyxnat.tests import skip_if_no_network
@skip_if_no_network
def test_schemasinspector():
    si = SchemasInspector(central)
    si()