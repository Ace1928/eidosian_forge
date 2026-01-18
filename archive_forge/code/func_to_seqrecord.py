import re
import warnings
from Bio.Align import Alignment, MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqFeature import SeqFeature, SimpleLocation
from Bio.SeqRecord import SeqRecord
from Bio import BiopythonWarning
from Bio.Phylo import BaseTree
def to_seqrecord(self):
    """Create a SeqRecord object from this Sequence instance.

        The seqrecord.annotations dictionary is packed like so::

            { # Sequence attributes with no SeqRecord equivalent:
              'id_ref': self.id_ref,
              'id_source': self.id_source,
              'location': self.location,
              'uri': { 'value': self.uri.value,
                              'desc': self.uri.desc,
                              'type': self.uri.type },
              # Sequence.annotations attribute (list of Annotations)
              'annotations': [{'ref': ann.ref,
                               'source': ann.source,
                               'evidence': ann.evidence,
                               'type': ann.type,
                               'confidence': [ann.confidence.value,
                                              ann.confidence.type],
                               'properties': [{'value': prop.value,
                                                'ref': prop.ref,
                                                'applies_to': prop.applies_to,
                                                'datatype': prop.datatype,
                                                'unit': prop.unit,
                                                'id_ref': prop.id_ref}
                                               for prop in ann.properties],
                              } for ann in self.annotations],
            }

        """

    def clean_dict(dct):
        """Remove None-valued items from a dictionary."""
        return {key: val for key, val in dct.items() if val is not None}
    seqrec = SeqRecord(Seq(self.mol_seq.value), **clean_dict({'id': str(self.accession), 'name': self.symbol, 'description': self.name}))
    if self.domain_architecture:
        seqrec.features = [dom.to_seqfeature() for dom in self.domain_architecture.domains]
    if self.type == 'dna':
        molecule_type = 'DNA'
    elif self.type == 'rna':
        molecule_type = 'RNA'
    elif self.type == 'protein':
        molecule_type = 'protein'
    else:
        molecule_type = None
    seqrec.annotations = clean_dict({'id_ref': self.id_ref, 'id_source': self.id_source, 'location': self.location, 'uri': self.uri and clean_dict({'value': self.uri.value, 'desc': self.uri.desc, 'type': self.uri.type}), 'molecule_type': molecule_type, 'annotations': self.annotations and [clean_dict({'ref': ann.ref, 'source': ann.source, 'evidence': ann.evidence, 'type': ann.type, 'confidence': ann.confidence and [ann.confidence.value, ann.confidence.type], 'properties': [clean_dict({'value': prop.value, 'ref': prop.ref, 'applies_to': prop.applies_to, 'datatype': prop.datatype, 'unit': prop.unit, 'id_ref': prop.id_ref}) for prop in ann.properties]}) for ann in self.annotations]})
    return seqrec