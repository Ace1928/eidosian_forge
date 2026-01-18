import re
import textwrap
from collections import defaultdict
from nltk.corpus.reader.xmldocs import XMLCorpusReader
def vnclass(self, fileid_or_classid):
    """Returns VerbNet class ElementTree

        Return an ElementTree containing the xml for the specified
        VerbNet class.

        :param fileid_or_classid: An identifier specifying which class
            should be returned.  Can be a file identifier (such as
            ``'put-9.1.xml'``), or a VerbNet class identifier (such as
            ``'put-9.1'``) or a short VerbNet class identifier (such as
            ``'9.1'``).
        """
    if fileid_or_classid in self._fileids:
        return self.xml(fileid_or_classid)
    classid = self.longid(fileid_or_classid)
    if classid in self._class_to_fileid:
        fileid = self._class_to_fileid[self.longid(classid)]
        tree = self.xml(fileid)
        if classid == tree.get('ID'):
            return tree
        else:
            for subclass in tree.findall('.//VNSUBCLASS'):
                if classid == subclass.get('ID'):
                    return subclass
            else:
                assert False
    else:
        raise ValueError(f'Unknown identifier {fileid_or_classid}')