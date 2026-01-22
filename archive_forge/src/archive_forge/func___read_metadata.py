import xml.etree.ElementTree as ET
from Bio.motifs import meme
def __read_metadata(record, xml_tree):
    record.version = xml_tree.getroot().get('version')
    record.database = xml_tree.find('sequence_dbs').find('sequence_db').get('source')
    record.alphabet = xml_tree.find('alphabet').get('name')
    record.strand_handling = xml_tree.find('settings').get('strand_handling')
    for i, motif_tree in enumerate(xml_tree.find('motifs').findall('motif')):
        motif = meme.Motif(record.alphabet)
        motif.name = str(i + 1)
        motif.id = motif_tree.get('id')
        motif.alt_id = motif_tree.get('alt')
        motif.length = int(motif_tree.get('length'))
        record.append(motif)