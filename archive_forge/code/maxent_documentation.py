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

        Construct and return new feature encoding, based on a given
        training corpus ``train_toks``.  See the class description
        ``TypedMaxentFeatureEncoding`` for a description of the
        joint-features that will be included in this encoding.

        Note: recognized feature values types are (int, float), over
        types are interpreted as regular binary features.

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
        