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
def _buildrelationindex(self):
    self._freltyp_idx = {}
    self._frel_idx = {}
    self._frel_f_idx = defaultdict(set)
    self._ferel_idx = {}
    with XMLCorpusView(self.abspath('frRelation.xml'), 'frameRelations/frameRelationType', self._handle_framerelationtype_elt) as view:
        for freltyp in view:
            self._freltyp_idx[freltyp.ID] = freltyp
            for frel in freltyp.frameRelations:
                supF = frel.superFrame = frel[freltyp.superFrameName] = Future((lambda fID: lambda: self.frame_by_id(fID))(frel.supID))
                subF = frel.subFrame = frel[freltyp.subFrameName] = Future((lambda fID: lambda: self.frame_by_id(fID))(frel.subID))
                self._frel_idx[frel.ID] = frel
                self._frel_f_idx[frel.supID].add(frel.ID)
                self._frel_f_idx[frel.subID].add(frel.ID)
                for ferel in frel.feRelations:
                    ferel.superFrame = supF
                    ferel.subFrame = subF
                    ferel.superFE = Future((lambda fer: lambda: fer.superFrame.FE[fer.superFEName])(ferel))
                    ferel.subFE = Future((lambda fer: lambda: fer.subFrame.FE[fer.subFEName])(ferel))
                    self._ferel_idx[ferel.ID] = ferel