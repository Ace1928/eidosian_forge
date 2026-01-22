from xml.etree import ElementTree
from xml.parsers.expat import errors
from Bio import SeqFeature
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
def _parse_evidence(element):
    for k, v in element.attrib.items():
        ann_key = k
        append_to_annotations(ann_key, v)