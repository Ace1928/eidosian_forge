import functools
import os
import re
import tempfile
from nltk.corpus.reader.util import concat
from nltk.corpus.reader.xmldocs import XMLCorpusReader, XMLCorpusView
class NKJPCorpus_Header_View(XMLCorpusView):

    def __init__(self, filename, **kwargs):
        """
        HEADER_MODE
        A stream backed corpus view specialized for use with
        header.xml files in NKJP corpus.
        """
        self.tagspec = '.*/sourceDesc$'
        XMLCorpusView.__init__(self, filename + 'header.xml', self.tagspec)

    def handle_query(self):
        self._open()
        header = []
        while True:
            segm = XMLCorpusView.read_block(self, self._stream)
            if len(segm) == 0:
                break
            header.extend(segm)
        self.close()
        return header

    def handle_elt(self, elt, context):
        titles = elt.findall('bibl/title')
        title = []
        if titles:
            title = '\n'.join((title.text.strip() for title in titles))
        authors = elt.findall('bibl/author')
        author = []
        if authors:
            author = '\n'.join((author.text.strip() for author in authors))
        dates = elt.findall('bibl/date')
        date = []
        if dates:
            date = '\n'.join((date.text.strip() for date in dates))
        publishers = elt.findall('bibl/publisher')
        publisher = []
        if publishers:
            publisher = '\n'.join((publisher.text.strip() for publisher in publishers))
        idnos = elt.findall('bibl/idno')
        idno = []
        if idnos:
            idno = '\n'.join((idno.text.strip() for idno in idnos))
        notes = elt.findall('bibl/note')
        note = []
        if notes:
            note = '\n'.join((note.text.strip() for note in notes))
        return {'title': title, 'author': author, 'date': date, 'publisher': publisher, 'idno': idno, 'note': note}