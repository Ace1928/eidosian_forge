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
class GISEncoding(BinaryMaxentFeatureEncoding):
    """
    A binary feature encoding which adds one new joint-feature to the
    joint-features defined by ``BinaryMaxentFeatureEncoding``: a
    correction feature, whose value is chosen to ensure that the
    sparse vector always sums to a constant non-negative number.  This
    new feature is used to ensure two preconditions for the GIS
    training algorithm:

      - At least one feature vector index must be nonzero for every
        token.
      - The feature vector must sum to a constant non-negative number
        for every token.
    """

    def __init__(self, labels, mapping, unseen_features=False, alwayson_features=False, C=None):
        """
        :param C: The correction constant.  The value of the correction
            feature is based on this value.  In particular, its value is
            ``C - sum([v for (f,v) in encoding])``.
        :seealso: ``BinaryMaxentFeatureEncoding.__init__``
        """
        BinaryMaxentFeatureEncoding.__init__(self, labels, mapping, unseen_features, alwayson_features)
        if C is None:
            C = len({fname for fname, fval, label in mapping}) + 1
        self._C = C

    @property
    def C(self):
        """The non-negative constant that all encoded feature vectors
        will sum to."""
        return self._C

    def encode(self, featureset, label):
        encoding = BinaryMaxentFeatureEncoding.encode(self, featureset, label)
        base_length = BinaryMaxentFeatureEncoding.length(self)
        total = sum((v for f, v in encoding))
        if total >= self._C:
            raise ValueError('Correction feature is not high enough!')
        encoding.append((base_length, self._C - total))
        return encoding

    def length(self):
        return BinaryMaxentFeatureEncoding.length(self) + 1

    def describe(self, f_id):
        if f_id == BinaryMaxentFeatureEncoding.length(self):
            return 'Correction feature (%s)' % self._C
        else:
            return BinaryMaxentFeatureEncoding.describe(self, f_id)