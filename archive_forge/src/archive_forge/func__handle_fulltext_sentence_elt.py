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
def _handle_fulltext_sentence_elt(self, elt):
    """Load information from the given 'sentence' element. Each
        'sentence' element contains a "text" and "annotationSet" sub
        elements."""
    info = self._load_xml_attributes(AttrDict(), elt)
    info['_type'] = 'fulltext_sentence'
    info['annotationSet'] = []
    info['targets'] = []
    target_spans = set()
    info['_ascii'] = types.MethodType(_annotation_ascii, info)
    info['text'] = ''
    for sub in elt:
        if sub.tag.endswith('text'):
            info['text'] = self._strip_tags(sub.text)
        elif sub.tag.endswith('annotationSet'):
            a = self._handle_fulltextannotationset_elt(sub, is_pos=len(info['annotationSet']) == 0)
            if 'cxnID' in a:
                continue
            a.sent = info
            a.text = info.text
            info['annotationSet'].append(a)
            if 'Target' in a:
                for tspan in a.Target:
                    if tspan in target_spans:
                        self._warn('Duplicate target span "{}"'.format(info.text[slice(*tspan)]), tspan, 'in sentence', info['ID'], info.text)
                    else:
                        target_spans.add(tspan)
                info['targets'].append((a.Target, a.luName, a.frameName))
    assert info['annotationSet'][0].status == 'UNANN'
    info['POS'] = info['annotationSet'][0].POS
    info['POS_tagset'] = info['annotationSet'][0].POS_tagset
    return info