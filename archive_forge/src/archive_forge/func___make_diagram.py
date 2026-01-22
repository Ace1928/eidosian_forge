import xml.etree.ElementTree as ET
from Bio.motifs import meme
def __make_diagram(record, sequence_tree):
    """Make diagram string found in text file based on motif hit info."""
    sequence_length = int(sequence_tree.get('length'))
    hit_eles, hit_motifs, gaps = ([], [], [])
    for seg_tree in sequence_tree.findall('seg'):
        for hit_ele in seg_tree.findall('hit'):
            hit_pos = int(hit_ele.get('pos'))
            if not hit_eles:
                gap = hit_pos - 1
            else:
                gap = hit_pos - int(hit_eles[-1].get('pos')) - hit_motifs[-1].length
            gaps.append(gap)
            hit_motifs.append(record[int(hit_ele.get('idx'))])
            hit_eles.append(hit_ele)
    if not hit_eles:
        return str(sequence_length)
    if record.strand_handling == 'combine':
        motif_strs = [f'[{('-' if hit_ele.get('rc') == 'y' else '+')}{hit_motif.name}]' for hit_ele, hit_motif in zip(hit_eles, hit_motifs)]
    elif record.strand_handling == 'unstranded':
        motif_strs = [f'[{hit_motif.name}]' for hit_ele, hit_motif in zip(hit_eles, hit_motifs)]
    else:
        raise Exception(f'Strand handling option {record.strand_handling} not parsable')
    tail_length = sequence_length - int(hit_eles[-1].get('pos')) - hit_motifs[-1].length + 1
    motifs_with_gaps = [str(s) for pair in zip(gaps, motif_strs) for s in pair] + [str(tail_length)]
    motifs_with_gaps = [s for s in motifs_with_gaps if s != '0']
    return '-'.join(motifs_with_gaps)