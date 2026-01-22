import itertools
import os
import re
import sys
import textwrap
import types
from collections import OrderedDict, defaultdict
from itertools import zip_longest
from operator import itemgetter
from pprint import pprint
from nltk.corpus.reader import XMLCorpusReader, XMLCorpusView
from nltk.util import LazyConcatenation, LazyIteratorList, LazyMap
def _handle_framelexunit_elt(self, elt):
    """Load the lexical unit info from an xml element in a frame's xml file."""
    luinfo = AttrDict()
    luinfo['_type'] = 'lu'
    luinfo = self._load_xml_attributes(luinfo, elt)
    luinfo['definition'] = ''
    luinfo['definitionMarkup'] = ''
    luinfo['sentenceCount'] = PrettyDict()
    luinfo['lexemes'] = PrettyList()
    luinfo['semTypes'] = PrettyList()
    for sub in elt:
        if sub.tag.endswith('definition'):
            luinfo['definitionMarkup'] = sub.text
            luinfo['definition'] = self._strip_tags(sub.text)
        elif sub.tag.endswith('sentenceCount'):
            luinfo['sentenceCount'] = self._load_xml_attributes(PrettyDict(), sub)
        elif sub.tag.endswith('lexeme'):
            lexemeinfo = self._load_xml_attributes(PrettyDict(), sub)
            if not isinstance(lexemeinfo.name, str):
                lexemeinfo.name = str(lexemeinfo.name)
            luinfo['lexemes'].append(lexemeinfo)
        elif sub.tag.endswith('semType'):
            semtypeinfo = self._load_xml_attributes(PrettyDict(), sub)
            luinfo['semTypes'].append(self.semtype(semtypeinfo.ID))
    luinfo['lexemes'].sort(key=lambda x: x.order)
    return luinfo