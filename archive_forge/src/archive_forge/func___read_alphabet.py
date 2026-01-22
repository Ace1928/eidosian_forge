import xml.etree.ElementTree as ET
from Bio import Align
from Bio import Seq
from Bio import motifs
def __read_alphabet(record, xml_tree):
    alphabet_tree = xml_tree.find('training_set').find('letter_frequencies').find('alphabet_array')
    for value in alphabet_tree.findall('value'):
        record.alphabet += value.get('letter_id')