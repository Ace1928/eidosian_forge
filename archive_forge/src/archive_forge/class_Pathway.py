import time
from itertools import chain
from xml.dom import minidom
import xml.etree.ElementTree as ET
class Pathway:
    """Represents a KGML pathway from KEGG.

    Specifies graph information for the pathway map, as described in
    release KGML v0.7.2 (http://www.kegg.jp/kegg/xml/docs/)

    Attributes:
     - name - KEGGID of the pathway map
     - org - ko/ec/[org prefix]
     - number - map number (integer)
     - title - the map title
     - image - URL of the image map for the pathway
     - link - URL of information about the pathway
     - entries - Dictionary of entries in the pathway, keyed by node ID
     - reactions - Set of reactions in the pathway

    The name attribute has a restricted format, so we make it a property and
    enforce the formatting.

    The Pathway object is the only allowed route for adding/removing
    Entry, Reaction, or Relation elements.

    Entries are held in a dictionary and keyed by the node ID for the
    pathway graph - this allows for ready access via the Reaction/Relation
    etc. elements.  Entries must be added before reference by any other
    element.

    Reactions are held in a dictionary, keyed by node ID for the path.
    The elements referred to in the reaction must be added before the
    reaction itself.

    """

    def __init__(self):
        """Initialize the class."""
        self._name = ''
        self.org = ''
        self._number = None
        self.title = ''
        self.image = ''
        self.link = ''
        self.entries = {}
        self._reactions = {}
        self._relations = set()

    def get_KGML(self):
        """Return the pathway as a string in prettified KGML format."""
        header = '\n'.join(['<?xml version="1.0"?>', '<!DOCTYPE pathway SYSTEM "http://www.genome.jp/kegg/xml/KGML_v0.7.2_.dtd">', f'<!-- Created by KGML_Pathway.py {time.asctime()} -->'])
        rough_xml = header + ET.tostring(self.element, 'utf-8').decode()
        reparsed = minidom.parseString(rough_xml)
        return reparsed.toprettyxml(indent='  ')

    def add_entry(self, entry):
        """Add an Entry element to the pathway."""
        if not isinstance(entry.id, int):
            raise TypeError(f'Node ID must be an integer, got {type(entry.id)} ({entry.id})')
        entry._pathway = self
        self.entries[entry.id] = entry

    def remove_entry(self, entry):
        """Remove an Entry element from the pathway."""
        if not isinstance(entry.id, int):
            raise TypeError(f'Node ID must be an integer, got {type(entry.id)} ({entry.id})')
        del self.entries[entry.id]

    def add_reaction(self, reaction):
        """Add a Reaction element to the pathway."""
        if not isinstance(reaction.id, int):
            raise ValueError(f'Node ID must be an integer, got {type(reaction.id)} ({reaction.id})')
        if reaction.id not in self.entries:
            raise ValueError('Reaction ID %d has no corresponding entry' % reaction.id)
        reaction._pathway = self
        self._reactions[reaction.id] = reaction

    def remove_reaction(self, reaction):
        """Remove a Reaction element from the pathway."""
        if not isinstance(reaction.id, int):
            raise TypeError(f'Node ID must be an integer, got {type(reaction.id)} ({reaction.id})')
        del self._reactions[reaction.id]

    def add_relation(self, relation):
        """Add a Relation element to the pathway."""
        relation._pathway = self
        self._relations.add(relation)

    def remove_relation(self, relation):
        """Remove a Relation element from the pathway."""
        self._relations.remove(relation)

    def __str__(self):
        """Return a readable summary description string."""
        outstr = [f'Pathway: {self.title}', f'KEGG ID: {self.name}', f'Image file: {self.image}', f'Organism: {self.org}', 'Entries: %d' % len(self.entries), 'Entry types:']
        for t in ['ortholog', 'enzyme', 'reaction', 'gene', 'group', 'compound', 'map']:
            etype = [e for e in self.entries.values() if e.type == t]
            if len(etype):
                outstr.append('\t%s: %d' % (t, len(etype)))
        return '\n'.join(outstr) + '\n'

    def _getname(self):
        return self._name

    def _setname(self, value):
        if not value.startswith('path:'):
            raise ValueError(f"Pathway name should begin with 'path:', got {value}")
        self._name = value

    def _delname(self):
        del self._name
    name = property(_getname, _setname, _delname, 'The KEGGID for the pathway map.')

    def _getnumber(self):
        return self._number

    def _setnumber(self, value):
        self._number = int(value)

    def _delnumber(self):
        del self._number
    number = property(_getnumber, _setnumber, _delnumber, 'The KEGG map number.')

    @property
    def compounds(self):
        """Get a list of entries of type compound."""
        return [e for e in self.entries.values() if e.type == 'compound']

    @property
    def maps(self):
        """Get a list of entries of type map."""
        return [e for e in self.entries.values() if e.type == 'map']

    @property
    def orthologs(self):
        """Get a list of entries of type ortholog."""
        return [e for e in self.entries.values() if e.type == 'ortholog']

    @property
    def genes(self):
        """Get a list of entries of type gene."""
        return [e for e in self.entries.values() if e.type == 'gene']

    @property
    def reactions(self):
        """Get a list of reactions in the pathway."""
        return self._reactions.values()

    @property
    def reaction_entries(self):
        """List of entries corresponding to each reaction in the pathway."""
        return [self.entries[i] for i in self._reactions]

    @property
    def relations(self):
        """Get a list of relations in the pathway."""
        return list(self._relations)

    @property
    def element(self):
        """Return the Pathway as a valid KGML element."""
        pathway = ET.Element('pathway')
        pathway.attrib = {'name': self._name, 'org': self.org, 'number': str(self._number), 'title': self.title, 'image': self.image, 'link': self.link}
        for eid, entry in sorted(self.entries.items()):
            pathway.append(entry.element)
        for relation in self._relations:
            pathway.append(relation.element)
        for eid, reaction in sorted(self._reactions.items()):
            pathway.append(reaction.element)
        return pathway

    @property
    def bounds(self):
        """Coordinate bounds for all Graphics elements in the Pathway.

        Returns the [(xmin, ymin), (xmax, ymax)] coordinates for all
        Graphics elements in the Pathway
        """
        xlist, ylist = ([], [])
        for b in [g.bounds for g in self.entries.values()]:
            xlist.extend([b[0][0], b[1][0]])
            ylist.extend([b[0][1], b[1][1]])
        return [(min(xlist), min(ylist)), (max(xlist), max(ylist))]