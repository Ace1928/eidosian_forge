import os
import tempfile
from collections import defaultdict
from nltk.classify.api import ClassifierI
from nltk.classify.megam import call_megam, parse_megam_weights, write_megam_file
from nltk.classify.tadm import call_tadm, parse_tadm_weights, write_tadm_file
from nltk.classify.util import CutoffChecker, accuracy, log_likelihood
from nltk.data import gzip_open_unicode
from nltk.probability import DictionaryProbDist
from nltk.util import OrderedDict
class TadmEventMaxentFeatureEncoding(BinaryMaxentFeatureEncoding):

    def __init__(self, labels, mapping, unseen_features=False, alwayson_features=False):
        self._mapping = OrderedDict(mapping)
        self._label_mapping = OrderedDict()
        BinaryMaxentFeatureEncoding.__init__(self, labels, self._mapping, unseen_features, alwayson_features)

    def encode(self, featureset, label):
        encoding = []
        for feature, value in featureset.items():
            if (feature, label) not in self._mapping:
                self._mapping[feature, label] = len(self._mapping)
            if value not in self._label_mapping:
                if not isinstance(value, int):
                    self._label_mapping[value] = len(self._label_mapping)
                else:
                    self._label_mapping[value] = value
            encoding.append((self._mapping[feature, label], self._label_mapping[value]))
        return encoding

    def labels(self):
        return self._labels

    def describe(self, fid):
        for feature, label in self._mapping:
            if self._mapping[feature, label] == fid:
                return (feature, label)

    def length(self):
        return len(self._mapping)

    @classmethod
    def train(cls, train_toks, count_cutoff=0, labels=None, **options):
        mapping = OrderedDict()
        if not labels:
            labels = []
        train_toks = list(train_toks)
        for featureset, label in train_toks:
            if label not in labels:
                labels.append(label)
        for featureset, label in train_toks:
            for label in labels:
                for feature in featureset:
                    if (feature, label) not in mapping:
                        mapping[feature, label] = len(mapping)
        return cls(labels, mapping, **options)