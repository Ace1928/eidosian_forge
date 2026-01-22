from xml.etree import ElementTree
from xml.parsers.expat import errors
from Bio import SeqFeature
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
def _parse_position(element, offset=0):
    try:
        position = int(element.attrib['position']) + offset
    except KeyError:
        position = None
    status = element.attrib.get('status', '')
    if status == 'unknown':
        assert position is None
        return SeqFeature.UnknownPosition()
    elif not status:
        return SeqFeature.ExactPosition(position)
    elif status == 'greater than':
        return SeqFeature.AfterPosition(position)
    elif status == 'less than':
        return SeqFeature.BeforePosition(position)
    elif status == 'uncertain':
        return SeqFeature.UncertainPosition(position)
    else:
        raise NotImplementedError(f'Position status {status!r}')