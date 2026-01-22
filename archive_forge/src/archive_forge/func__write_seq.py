from xml import sax
from xml.sax import handler
from xml.sax.saxutils import XMLGenerator
from xml.sax.xmlreader import AttributesImpl
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from .Interfaces import SequenceIterator
from .Interfaces import SequenceWriter
def _write_seq(self, record):
    """Write the sequence (PRIVATE).

        Note that SeqXML requires the molecule type to contain the term
        "DNA", "RNA", or "protein".
        """
    seq = bytes(record.seq)
    if not len(seq) > 0:
        raise ValueError('The sequence length should be greater than 0')
    molecule_type = record.annotations.get('molecule_type')
    if molecule_type is None:
        raise ValueError('molecule_type is not defined')
    elif 'DNA' in molecule_type:
        seqElem = 'DNAseq'
    elif 'RNA' in molecule_type:
        seqElem = 'RNAseq'
    elif 'protein' in molecule_type:
        seqElem = 'AAseq'
    else:
        raise ValueError(f"unknown molecule_type '{molecule_type}'")
    self.xml_generator.startElement(seqElem, AttributesImpl({}))
    self.xml_generator.characters(seq)
    self.xml_generator.endElement(seqElem)