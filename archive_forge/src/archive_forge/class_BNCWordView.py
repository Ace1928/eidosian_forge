from nltk.corpus.reader.util import concat
from nltk.corpus.reader.xmldocs import ElementTree, XMLCorpusReader, XMLCorpusView
class BNCWordView(XMLCorpusView):
    """
    A stream backed corpus view specialized for use with the BNC corpus.
    """
    tags_to_ignore = {'pb', 'gap', 'vocal', 'event', 'unclear', 'shift', 'pause', 'align'}
    'These tags are ignored. For their description refer to the\n    technical documentation, for example,\n    http://www.natcorp.ox.ac.uk/docs/URG/ref-vocal.html\n\n    '

    def __init__(self, fileid, sent, tag, strip_space, stem):
        """
        :param fileid: The name of the underlying file.
        :param sent: If true, include sentence bracketing.
        :param tag: The name of the tagset to use, or None for no tags.
        :param strip_space: If true, strip spaces from word tokens.
        :param stem: If true, then substitute stems for words.
        """
        if sent:
            tagspec = '.*/s'
        else:
            tagspec = '.*/s/(.*/)?(c|w)'
        self._sent = sent
        self._tag = tag
        self._strip_space = strip_space
        self._stem = stem
        self.title = None
        self.author = None
        self.editor = None
        self.resps = None
        XMLCorpusView.__init__(self, fileid, tagspec)
        self._open()
        self.read_block(self._stream, '.*/teiHeader$', self.handle_header)
        self.close()
        self._tag_context = {0: ()}

    def handle_header(self, elt, context):
        titles = elt.findall('titleStmt/title')
        if titles:
            self.title = '\n'.join((title.text.strip() for title in titles))
        authors = elt.findall('titleStmt/author')
        if authors:
            self.author = '\n'.join((author.text.strip() for author in authors))
        editors = elt.findall('titleStmt/editor')
        if editors:
            self.editor = '\n'.join((editor.text.strip() for editor in editors))
        resps = elt.findall('titleStmt/respStmt')
        if resps:
            self.resps = '\n\n'.join(('\n'.join((resp_elt.text.strip() for resp_elt in resp)) for resp in resps))

    def handle_elt(self, elt, context):
        if self._sent:
            return self.handle_sent(elt)
        else:
            return self.handle_word(elt)

    def handle_word(self, elt):
        word = elt.text
        if not word:
            word = ''
        if self._strip_space or self._stem:
            word = word.strip()
        if self._stem:
            word = elt.get('hw', word)
        if self._tag == 'c5':
            word = (word, elt.get('c5'))
        elif self._tag == 'pos':
            word = (word, elt.get('pos', elt.get('c5')))
        return word

    def handle_sent(self, elt):
        sent = []
        for child in elt:
            if child.tag in ('mw', 'hi', 'corr', 'trunc'):
                sent += [self.handle_word(w) for w in child]
            elif child.tag in ('w', 'c'):
                sent.append(self.handle_word(child))
            elif child.tag not in self.tags_to_ignore:
                raise ValueError('Unexpected element %s' % child.tag)
        return BNCSentence(elt.attrib['n'], sent)