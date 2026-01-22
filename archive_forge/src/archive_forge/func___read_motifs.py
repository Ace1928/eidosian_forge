import xml.etree.ElementTree as ET
from Bio import Align
from Bio import Seq
from Bio import motifs
def __read_motifs(record, xml_tree, sequence_id_name_map):
    for motif_tree in xml_tree.find('motifs').findall('motif'):
        instances = []
        for site_tree in motif_tree.find('contributing_sites').findall('contributing_site'):
            letters = [letter_ref.get('letter_id') for letter_ref in site_tree.find('site').findall('letter_ref')]
            sequence = ''.join(letters)
            instance = Instance(sequence)
            instance.motif_name = motif_tree.get('name')
            instance.sequence_id = site_tree.get('sequence_id')
            instance.sequence_name = sequence_id_name_map[instance.sequence_id]
            instance.start = int(site_tree.get('position')) + 1
            instance.pvalue = float(site_tree.get('pvalue'))
            instance.strand = __convert_strand(site_tree.get('strand'))
            instance.length = len(sequence)
            instances.append(instance)
        alignment = Align.Alignment(instances)
        motif = Motif(record.alphabet, alignment)
        motif.id = motif_tree.get('id')
        motif.name = motif_tree.get('name')
        motif.alt_id = motif_tree.get('alt')
        motif.length = int(motif_tree.get('width'))
        motif.num_occurrences = int(motif_tree.get('sites'))
        motif.evalue = float(motif_tree.get('e_value'))
        record.append(motif)