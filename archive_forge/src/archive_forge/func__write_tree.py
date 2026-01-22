from io import StringIO
from xml.dom import minidom
from xml.etree import ElementTree
from Bio.Phylo import NeXML
from ._cdao_owl import cdao_elements, cdao_namespaces, resolve_uri
def _write_tree(self, clade, tree, parent=None, rooted=False):
    """Recursively process tree, adding nodes and edges to Tree object (PRIVATE).

        Returns a set of all OTUs encountered.
        """
    tus = set()
    convert_uri = cdao_to_obo if self.cdao_to_obo else lambda s: s
    node_id = self.new_label('node')
    clade.node_id = node_id
    attrib = {'id': node_id, 'label': node_id}
    root = rooted and parent is None
    if root:
        attrib['root'] = 'true'
    if clade.name:
        tus.add(clade.name)
        attrib['otu'] = clade.name
    node = ElementTree.SubElement(tree, 'node', **attrib)
    if parent is not None:
        edge_id = self.new_label('edge')
        attrib = {'id': edge_id, 'source': parent.node_id, 'target': node_id, 'length': str(clade.branch_length), 'typeof': convert_uri('cdao:Edge')}
        try:
            confidence = clade.confidence
        except AttributeError:
            pass
        else:
            if confidence is not None:
                attrib.update({'property': convert_uri('cdao:has_Support_Value'), 'datatype': 'xsd:float', 'content': f'{confidence:1.2f}'})
        node = ElementTree.SubElement(tree, 'edge', **attrib)
    if not clade.is_terminal():
        for new_clade in clade.clades:
            tus.update(self._write_tree(new_clade, tree, parent=clade))
    del clade.node_id
    return tus