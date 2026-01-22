from GSL Biotech LLC.
from datetime import datetime
from re import sub
from struct import unpack
from xml.dom.minidom import parseString
from Bio.Seq import Seq
from Bio.SeqFeature import SimpleLocation
from Bio.SeqFeature import SeqFeature
from Bio.SeqRecord import SeqRecord
from .Interfaces import SequenceIterator
def _parse_features_packet(length, data, record):
    """Parse a sequence features packet.

    This packet stores sequence features (except primer binding sites,
    which are in a dedicated Primers packet). The data is a XML string
    starting with a 'Features' root node.
    """
    xml = parseString(data.decode('UTF-8'))
    for feature in xml.getElementsByTagName('Feature'):
        quals = {}
        type = _get_attribute_value(feature, 'type', default='misc_feature')
        strand = +1
        directionality = int(_get_attribute_value(feature, 'directionality', default='1'))
        if directionality == 2:
            strand = -1
        location = None
        subparts = []
        n_parts = 0
        for segment in feature.getElementsByTagName('Segment'):
            if _get_attribute_value(segment, 'type', 'standard') == 'gap':
                continue
            rng = _get_attribute_value(segment, 'range')
            n_parts += 1
            next_location = _parse_location(rng, strand, record)
            if not location:
                location = next_location
            elif strand == -1:
                location = next_location + location
            else:
                location = location + next_location
            name = _get_attribute_value(segment, 'name')
            if name:
                subparts.append([n_parts, name])
        if len(subparts) > 0:
            if strand == -1:
                subparts = reversed([[n_parts - i + 1, name] for i, name in subparts])
            quals['parts'] = [';'.join((f'{i}:{name}' for i, name in subparts))]
        if not location:
            raise ValueError('Missing feature location')
        for qualifier in feature.getElementsByTagName('Q'):
            qname = _get_attribute_value(qualifier, 'name', error='Missing qualifier name')
            qvalues = []
            for value in qualifier.getElementsByTagName('V'):
                if value.hasAttribute('text'):
                    qvalues.append(_decode(value.attributes['text'].value))
                elif value.hasAttribute('predef'):
                    qvalues.append(_decode(value.attributes['predef'].value))
                elif value.hasAttribute('int'):
                    qvalues.append(int(value.attributes['int'].value))
            quals[qname] = qvalues
        name = _get_attribute_value(feature, 'name')
        if name:
            if 'label' not in quals:
                quals['label'] = [name]
            elif name not in quals['label']:
                quals['name'] = [name]
        feature = SeqFeature(location, type=type, qualifiers=quals)
        record.features.append(feature)