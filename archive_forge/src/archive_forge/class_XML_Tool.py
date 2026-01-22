import functools
import os
import re
import tempfile
from nltk.corpus.reader.util import concat
from nltk.corpus.reader.xmldocs import XMLCorpusReader, XMLCorpusView
class XML_Tool:
    """
    Helper class creating xml file to one without references to nkjp: namespace.
    That's needed because the XMLCorpusView assumes that one can find short substrings
    of XML that are valid XML, which is not true if a namespace is declared at top level
    """

    def __init__(self, root, filename):
        self.read_file = os.path.join(root, filename)
        self.write_file = tempfile.NamedTemporaryFile(delete=False)

    def build_preprocessed_file(self):
        try:
            fr = open(self.read_file)
            fw = self.write_file
            line = ' '
            while len(line):
                line = fr.readline()
                x = re.split('nkjp:[^ ]* ', line)
                ret = ' '.join(x)
                x = re.split('<nkjp:paren>', ret)
                ret = ' '.join(x)
                x = re.split('</nkjp:paren>', ret)
                ret = ' '.join(x)
                x = re.split('<choice>', ret)
                ret = ' '.join(x)
                x = re.split('</choice>', ret)
                ret = ' '.join(x)
                fw.write(ret)
            fr.close()
            fw.close()
            return self.write_file.name
        except Exception as e:
            self.remove_preprocessed_file()
            raise Exception from e

    def remove_preprocessed_file(self):
        os.remove(self.write_file.name)