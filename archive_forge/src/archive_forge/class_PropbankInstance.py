import re
from functools import total_ordering
from xml.etree import ElementTree
from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
from nltk.internals import raise_unorderable_types
from nltk.tree import Tree
class PropbankInstance:

    def __init__(self, fileid, sentnum, wordnum, tagger, roleset, inflection, predicate, arguments, parse_corpus=None):
        self.fileid = fileid
        "The name of the file containing the parse tree for this\n        instance's sentence."
        self.sentnum = sentnum
        'The sentence number of this sentence within ``fileid``.\n        Indexing starts from zero.'
        self.wordnum = wordnum
        "The word number of this instance's predicate within its\n        containing sentence.  Word numbers are indexed starting from\n        zero, and include traces and other empty parse elements."
        self.tagger = tagger
        "An identifier for the tagger who tagged this instance; or\n        ``'gold'`` if this is an adjuticated instance."
        self.roleset = roleset
        "The name of the roleset used by this instance's predicate.\n        Use ``propbank.roleset() <PropbankCorpusReader.roleset>`` to\n        look up information about the roleset."
        self.inflection = inflection
        "A ``PropbankInflection`` object describing the inflection of\n        this instance's predicate."
        self.predicate = predicate
        "A ``PropbankTreePointer`` indicating the position of this\n        instance's predicate within its containing sentence."
        self.arguments = tuple(arguments)
        "A list of tuples (argloc, argid), specifying the location\n        and identifier for each of the predicate's argument in the\n        containing sentence.  Argument identifiers are strings such as\n        ``'ARG0'`` or ``'ARGM-TMP'``.  This list does *not* contain\n        the predicate."
        self.parse_corpus = parse_corpus
        'A corpus reader for the parse trees corresponding to the\n        instances in this propbank corpus.'

    @property
    def baseform(self):
        """The baseform of the predicate."""
        return self.roleset.split('.')[0]

    @property
    def sensenumber(self):
        """The sense number of the predicate."""
        return self.roleset.split('.')[1]

    @property
    def predid(self):
        """Identifier of the predicate."""
        return 'rel'

    def __repr__(self):
        return '<PropbankInstance: {}, sent {}, word {}>'.format(self.fileid, self.sentnum, self.wordnum)

    def __str__(self):
        s = '{} {} {} {} {} {}'.format(self.fileid, self.sentnum, self.wordnum, self.tagger, self.roleset, self.inflection)
        items = self.arguments + ((self.predicate, 'rel'),)
        for argloc, argid in sorted(items):
            s += f' {argloc}-{argid}'
        return s

    def _get_tree(self):
        if self.parse_corpus is None:
            return None
        if self.fileid not in self.parse_corpus.fileids():
            return None
        return self.parse_corpus.parsed_sents(self.fileid)[self.sentnum]
    tree = property(_get_tree, doc='\n        The parse tree corresponding to this instance, or None if\n        the corresponding tree is not available.')

    @staticmethod
    def parse(s, parse_fileid_xform=None, parse_corpus=None):
        pieces = s.split()
        if len(pieces) < 7:
            raise ValueError('Badly formatted propbank line: %r' % s)
        fileid, sentnum, wordnum, tagger, roleset, inflection = pieces[:6]
        rel = [p for p in pieces[6:] if p.endswith('-rel')]
        args = [p for p in pieces[6:] if not p.endswith('-rel')]
        if len(rel) != 1:
            raise ValueError('Badly formatted propbank line: %r' % s)
        if parse_fileid_xform is not None:
            fileid = parse_fileid_xform(fileid)
        sentnum = int(sentnum)
        wordnum = int(wordnum)
        inflection = PropbankInflection.parse(inflection)
        predicate = PropbankTreePointer.parse(rel[0][:-4])
        arguments = []
        for arg in args:
            argloc, argid = arg.split('-', 1)
            arguments.append((PropbankTreePointer.parse(argloc), argid))
        return PropbankInstance(fileid, sentnum, wordnum, tagger, roleset, inflection, predicate, arguments, parse_corpus)