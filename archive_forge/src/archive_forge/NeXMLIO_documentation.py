from io import StringIO
from xml.dom import minidom
from xml.etree import ElementTree
from Bio.Phylo import NeXML
from ._cdao_owl import cdao_elements, cdao_namespaces, resolve_uri
Recursively process tree, adding nodes and edges to Tree object (PRIVATE).

        Returns a set of all OTUs encountered.
        