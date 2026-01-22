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
def _handle_luannotationset_elt(self, elt, is_pos=False):
    """Load an annotation set from a sentence in an subcorpus of an LU"""
    info = self._load_xml_attributes(AttrDict(), elt)
    info['_type'] = 'posannotationset' if is_pos else 'luannotationset'
    info['layer'] = []
    info['_ascii'] = types.MethodType(_annotation_ascii, info)
    if 'cxnID' in info:
        return info
    for sub in elt:
        if sub.tag.endswith('layer'):
            l = self._handle_lulayer_elt(sub)
            if l is not None:
                overt = []
                ni = {}
                info['layer'].append(l)
                for lbl in l.label:
                    if 'start' in lbl:
                        thespan = (lbl.start, lbl.end + 1, lbl.name)
                        if l.name not in ('Sent', 'Other'):
                            assert thespan not in overt, (info.ID, l.name, thespan)
                        overt.append(thespan)
                    elif lbl.name in ni:
                        self._warn('FE with multiple NI entries:', lbl.name, ni[lbl.name], lbl.itype)
                    else:
                        ni[lbl.name] = lbl.itype
                overt = sorted(overt)
                if l.name == 'Target':
                    if not overt:
                        self._warn('Skipping empty Target layer in annotation set ID={}'.format(info.ID))
                        continue
                    assert all((lblname == 'Target' for i, j, lblname in overt))
                    if 'Target' in info:
                        self._warn('Annotation set {} has multiple Target layers'.format(info.ID))
                    else:
                        info['Target'] = [(i, j) for i, j, _ in overt]
                elif l.name == 'FE':
                    if l.rank == 1:
                        assert 'FE' not in info
                        info['FE'] = (overt, ni)
                    else:
                        assert 2 <= l.rank <= 3, l.rank
                        k = 'FE' + str(l.rank)
                        assert k not in info
                        info[k] = (overt, ni)
                elif l.name in ('GF', 'PT'):
                    assert l.rank == 1
                    info[l.name] = overt
                elif l.name in ('BNC', 'PENN'):
                    assert l.rank == 1
                    info['POS'] = overt
                    info['POS_tagset'] = l.name
                else:
                    if is_pos:
                        if l.name not in ('NER', 'WSL'):
                            self._warn('Unexpected layer in sentence annotationset:', l.name)
                    elif l.name not in ('Sent', 'Verb', 'Noun', 'Adj', 'Adv', 'Prep', 'Scon', 'Art', 'Other'):
                        self._warn('Unexpected layer in frame annotationset:', l.name)
                    info[l.name] = overt
    if not is_pos and 'cxnID' not in info:
        if 'Target' not in info:
            self._warn(f'Missing target in annotation set ID={info.ID}')
        assert 'FE' in info
        if 'FE3' in info:
            assert 'FE2' in info
    return info