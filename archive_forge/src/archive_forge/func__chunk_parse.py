import codecs
import re
from io import StringIO
from xml.etree.ElementTree import Element, ElementTree, SubElement, TreeBuilder
from nltk.data import PathPointer, find
def _chunk_parse(self, grammar=None, root_label='record', trace=0, **kwargs):
    """
        Returns an element tree structure corresponding to a toolbox data file
        parsed according to the chunk grammar.

        :type grammar: str
        :param grammar: Contains the chunking rules used to parse the
            database.  See ``chunk.RegExp`` for documentation.
        :type root_label: str
        :param root_label: The node value that should be used for the
            top node of the chunk structure.
        :type trace: int
        :param trace: The level of tracing that should be used when
            parsing a text.  ``0`` will generate no tracing output;
            ``1`` will generate normal tracing output; and ``2`` or
            higher will generate verbose tracing output.
        :type kwargs: dict
        :param kwargs: Keyword arguments passed to ``toolbox.StandardFormat.fields()``
        :rtype: ElementTree._ElementInterface
        """
    from nltk import chunk
    from nltk.tree import Tree
    cp = chunk.RegexpParser(grammar, root_label=root_label, trace=trace)
    db = self.parse(**kwargs)
    tb_etree = Element('toolbox_data')
    header = db.find('header')
    tb_etree.append(header)
    for record in db.findall('record'):
        parsed = cp.parse([(elem.text, elem.tag) for elem in record])
        tb_etree.append(self._tree2etree(parsed))
    return tb_etree