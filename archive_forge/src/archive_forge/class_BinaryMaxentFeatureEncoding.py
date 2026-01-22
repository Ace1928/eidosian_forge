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
class BinaryMaxentFeatureEncoding(MaxentFeatureEncodingI):
    """
    A feature encoding that generates vectors containing a binary
    joint-features of the form:

    |  joint_feat(fs, l) = { 1 if (fs[fname] == fval) and (l == label)
    |                      {
    |                      { 0 otherwise

    Where ``fname`` is the name of an input-feature, ``fval`` is a value
    for that input-feature, and ``label`` is a label.

    Typically, these features are constructed based on a training
    corpus, using the ``train()`` method.  This method will create one
    feature for each combination of ``fname``, ``fval``, and ``label``
    that occurs at least once in the training corpus.

    The ``unseen_features`` parameter can be used to add "unseen-value
    features", which are used whenever an input feature has a value
    that was not encountered in the training corpus.  These features
    have the form:

    |  joint_feat(fs, l) = { 1 if is_unseen(fname, fs[fname])
    |                      {      and l == label
    |                      {
    |                      { 0 otherwise

    Where ``is_unseen(fname, fval)`` is true if the encoding does not
    contain any joint features that are true when ``fs[fname]==fval``.

    The ``alwayson_features`` parameter can be used to add "always-on
    features", which have the form::

    |  joint_feat(fs, l) = { 1 if (l == label)
    |                      {
    |                      { 0 otherwise

    These always-on features allow the maxent model to directly model
    the prior probabilities of each label.
    """

    def __init__(self, labels, mapping, unseen_features=False, alwayson_features=False):
        """
        :param labels: A list of the "known labels" for this encoding.

        :param mapping: A dictionary mapping from ``(fname,fval,label)``
            tuples to corresponding joint-feature indexes.  These
            indexes must be the set of integers from 0...len(mapping).
            If ``mapping[fname,fval,label]=id``, then
            ``self.encode(..., fname:fval, ..., label)[id]`` is 1;
            otherwise, it is 0.

        :param unseen_features: If true, then include unseen value
           features in the generated joint-feature vectors.

        :param alwayson_features: If true, then include always-on
           features in the generated joint-feature vectors.
        """
        if set(mapping.values()) != set(range(len(mapping))):
            raise ValueError('Mapping values must be exactly the set of integers from 0...len(mapping)')
        self._labels = list(labels)
        'A list of attested labels.'
        self._mapping = mapping
        'dict mapping from (fname,fval,label) -> fid'
        self._length = len(mapping)
        'The length of generated joint feature vectors.'
        self._alwayson = None
        'dict mapping from label -> fid'
        self._unseen = None
        'dict mapping from fname -> fid'
        if alwayson_features:
            self._alwayson = {label: i + self._length for i, label in enumerate(labels)}
            self._length += len(self._alwayson)
        if unseen_features:
            fnames = {fname for fname, fval, label in mapping}
            self._unseen = {fname: i + self._length for i, fname in enumerate(fnames)}
            self._length += len(fnames)

    def encode(self, featureset, label):
        encoding = []
        for fname, fval in featureset.items():
            if (fname, fval, label) in self._mapping:
                encoding.append((self._mapping[fname, fval, label], 1))
            elif self._unseen:
                for label2 in self._labels:
                    if (fname, fval, label2) in self._mapping:
                        break
                else:
                    if fname in self._unseen:
                        encoding.append((self._unseen[fname], 1))
        if self._alwayson and label in self._alwayson:
            encoding.append((self._alwayson[label], 1))
        return encoding

    def describe(self, f_id):
        if not isinstance(f_id, int):
            raise TypeError('describe() expected an int')
        try:
            self._inv_mapping
        except AttributeError:
            self._inv_mapping = [-1] * len(self._mapping)
            for info, i in self._mapping.items():
                self._inv_mapping[i] = info
        if f_id < len(self._mapping):
            fname, fval, label = self._inv_mapping[f_id]
            return f'{fname}=={fval!r} and label is {label!r}'
        elif self._alwayson and f_id in self._alwayson.values():
            for label, f_id2 in self._alwayson.items():
                if f_id == f_id2:
                    return 'label is %r' % label
        elif self._unseen and f_id in self._unseen.values():
            for fname, f_id2 in self._unseen.items():
                if f_id == f_id2:
                    return '%s is unseen' % fname
        else:
            raise ValueError('Bad feature id')

    def labels(self):
        return self._labels

    def length(self):
        return self._length

    @classmethod
    def train(cls, train_toks, count_cutoff=0, labels=None, **options):
        """
        Construct and return new feature encoding, based on a given
        training corpus ``train_toks``.  See the class description
        ``BinaryMaxentFeatureEncoding`` for a description of the
        joint-features that will be included in this encoding.

        :type train_toks: list(tuple(dict, str))
        :param train_toks: Training data, represented as a list of
            pairs, the first member of which is a feature dictionary,
            and the second of which is a classification label.

        :type count_cutoff: int
        :param count_cutoff: A cutoff value that is used to discard
            rare joint-features.  If a joint-feature's value is 1
            fewer than ``count_cutoff`` times in the training corpus,
            then that joint-feature is not included in the generated
            encoding.

        :type labels: list
        :param labels: A list of labels that should be used by the
            classifier.  If not specified, then the set of labels
            attested in ``train_toks`` will be used.

        :param options: Extra parameters for the constructor, such as
            ``unseen_features`` and ``alwayson_features``.
        """
        mapping = {}
        seen_labels = set()
        count = defaultdict(int)
        for tok, label in train_toks:
            if labels and label not in labels:
                raise ValueError('Unexpected label %s' % label)
            seen_labels.add(label)
            for fname, fval in tok.items():
                count[fname, fval] += 1
                if count[fname, fval] >= count_cutoff:
                    if (fname, fval, label) not in mapping:
                        mapping[fname, fval, label] = len(mapping)
        if labels is None:
            labels = seen_labels
        return cls(labels, mapping, **options)