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
class MaxentClassifier(ClassifierI):
    """
    A maximum entropy classifier (also known as a "conditional
    exponential classifier").  This classifier is parameterized by a
    set of "weights", which are used to combine the joint-features
    that are generated from a featureset by an "encoding".  In
    particular, the encoding maps each ``(featureset, label)`` pair to
    a vector.  The probability of each label is then computed using
    the following equation::

                                dotprod(weights, encode(fs,label))
      prob(fs|label) = ---------------------------------------------------
                       sum(dotprod(weights, encode(fs,l)) for l in labels)

    Where ``dotprod`` is the dot product::

      dotprod(a,b) = sum(x*y for (x,y) in zip(a,b))
    """

    def __init__(self, encoding, weights, logarithmic=True):
        """
        Construct a new maxent classifier model.  Typically, new
        classifier models are created using the ``train()`` method.

        :type encoding: MaxentFeatureEncodingI
        :param encoding: An encoding that is used to convert the
            featuresets that are given to the ``classify`` method into
            joint-feature vectors, which are used by the maxent
            classifier model.

        :type weights: list of float
        :param weights:  The feature weight vector for this classifier.

        :type logarithmic: bool
        :param logarithmic: If false, then use non-logarithmic weights.
        """
        self._encoding = encoding
        self._weights = weights
        self._logarithmic = logarithmic
        assert encoding.length() == len(weights)

    def labels(self):
        return self._encoding.labels()

    def set_weights(self, new_weights):
        """
        Set the feature weight vector for this classifier.
        :param new_weights: The new feature weight vector.
        :type new_weights: list of float
        """
        self._weights = new_weights
        assert self._encoding.length() == len(new_weights)

    def weights(self):
        """
        :return: The feature weight vector for this classifier.
        :rtype: list of float
        """
        return self._weights

    def classify(self, featureset):
        return self.prob_classify(featureset).max()

    def prob_classify(self, featureset):
        prob_dict = {}
        for label in self._encoding.labels():
            feature_vector = self._encoding.encode(featureset, label)
            if self._logarithmic:
                total = 0.0
                for f_id, f_val in feature_vector:
                    total += self._weights[f_id] * f_val
                prob_dict[label] = total
            else:
                prod = 1.0
                for f_id, f_val in feature_vector:
                    prod *= self._weights[f_id] ** f_val
                prob_dict[label] = prod
        return DictionaryProbDist(prob_dict, log=self._logarithmic, normalize=True)

    def explain(self, featureset, columns=4):
        """
        Print a table showing the effect of each of the features in
        the given feature set, and how they combine to determine the
        probabilities of each label for that featureset.
        """
        descr_width = 50
        TEMPLATE = '  %-' + str(descr_width - 2) + 's%s%8.3f'
        pdist = self.prob_classify(featureset)
        labels = sorted(pdist.samples(), key=pdist.prob, reverse=True)
        labels = labels[:columns]
        print('  Feature'.ljust(descr_width) + ''.join(('%8s' % ('%s' % l)[:7] for l in labels)))
        print('  ' + '-' * (descr_width - 2 + 8 * len(labels)))
        sums = defaultdict(int)
        for i, label in enumerate(labels):
            feature_vector = self._encoding.encode(featureset, label)
            feature_vector.sort(key=lambda fid__: abs(self._weights[fid__[0]]), reverse=True)
            for f_id, f_val in feature_vector:
                if self._logarithmic:
                    score = self._weights[f_id] * f_val
                else:
                    score = self._weights[f_id] ** f_val
                descr = self._encoding.describe(f_id)
                descr = descr.split(' and label is ')[0]
                descr += ' (%s)' % f_val
                if len(descr) > 47:
                    descr = descr[:44] + '...'
                print(TEMPLATE % (descr, i * 8 * ' ', score))
                sums[label] += score
        print('  ' + '-' * (descr_width - 1 + 8 * len(labels)))
        print('  TOTAL:'.ljust(descr_width) + ''.join(('%8.3f' % sums[l] for l in labels)))
        print('  PROBS:'.ljust(descr_width) + ''.join(('%8.3f' % pdist.prob(l) for l in labels)))

    def most_informative_features(self, n=10):
        """
        Generates the ranked list of informative features from most to least.
        """
        if hasattr(self, '_most_informative_features'):
            return self._most_informative_features[:n]
        else:
            self._most_informative_features = sorted(list(range(len(self._weights))), key=lambda fid: abs(self._weights[fid]), reverse=True)
            return self._most_informative_features[:n]

    def show_most_informative_features(self, n=10, show='all'):
        """
        :param show: all, neg, or pos (for negative-only or positive-only)
        :type show: str
        :param n: The no. of top features
        :type n: int
        """
        fids = self.most_informative_features(None)
        if show == 'pos':
            fids = [fid for fid in fids if self._weights[fid] > 0]
        elif show == 'neg':
            fids = [fid for fid in fids if self._weights[fid] < 0]
        for fid in fids[:n]:
            print(f'{self._weights[fid]:8.3f} {self._encoding.describe(fid)}')

    def __repr__(self):
        return '<ConditionalExponentialClassifier: %d labels, %d features>' % (len(self._encoding.labels()), self._encoding.length())
    ALGORITHMS = ['GIS', 'IIS', 'MEGAM', 'TADM']

    @classmethod
    def train(cls, train_toks, algorithm=None, trace=3, encoding=None, labels=None, gaussian_prior_sigma=0, **cutoffs):
        """
        Train a new maxent classifier based on the given corpus of
        training samples.  This classifier will have its weights
        chosen to maximize entropy while remaining empirically
        consistent with the training corpus.

        :rtype: MaxentClassifier
        :return: The new maxent classifier

        :type train_toks: list
        :param train_toks: Training data, represented as a list of
            pairs, the first member of which is a featureset,
            and the second of which is a classification label.

        :type algorithm: str
        :param algorithm: A case-insensitive string, specifying which
            algorithm should be used to train the classifier.  The
            following algorithms are currently available.

            - Iterative Scaling Methods: Generalized Iterative Scaling (``'GIS'``),
              Improved Iterative Scaling (``'IIS'``)
            - External Libraries (requiring megam):
              LM-BFGS algorithm, with training performed by Megam (``'megam'``)

            The default algorithm is ``'IIS'``.

        :type trace: int
        :param trace: The level of diagnostic tracing output to produce.
            Higher values produce more verbose output.
        :type encoding: MaxentFeatureEncodingI
        :param encoding: A feature encoding, used to convert featuresets
            into feature vectors.  If none is specified, then a
            ``BinaryMaxentFeatureEncoding`` will be built based on the
            features that are attested in the training corpus.
        :type labels: list(str)
        :param labels: The set of possible labels.  If none is given, then
            the set of all labels attested in the training data will be
            used instead.
        :param gaussian_prior_sigma: The sigma value for a gaussian
            prior on model weights.  Currently, this is supported by
            ``megam``. For other algorithms, its value is ignored.
        :param cutoffs: Arguments specifying various conditions under
            which the training should be halted.  (Some of the cutoff
            conditions are not supported by some algorithms.)

            - ``max_iter=v``: Terminate after ``v`` iterations.
            - ``min_ll=v``: Terminate after the negative average
              log-likelihood drops under ``v``.
            - ``min_lldelta=v``: Terminate if a single iteration improves
              log likelihood by less than ``v``.
        """
        if algorithm is None:
            algorithm = 'iis'
        for key in cutoffs:
            if key not in ('max_iter', 'min_ll', 'min_lldelta', 'max_acc', 'min_accdelta', 'count_cutoff', 'norm', 'explicit', 'bernoulli'):
                raise TypeError('Unexpected keyword arg %r' % key)
        algorithm = algorithm.lower()
        if algorithm == 'iis':
            return train_maxent_classifier_with_iis(train_toks, trace, encoding, labels, **cutoffs)
        elif algorithm == 'gis':
            return train_maxent_classifier_with_gis(train_toks, trace, encoding, labels, **cutoffs)
        elif algorithm == 'megam':
            return train_maxent_classifier_with_megam(train_toks, trace, encoding, labels, gaussian_prior_sigma, **cutoffs)
        elif algorithm == 'tadm':
            kwargs = cutoffs
            kwargs['trace'] = trace
            kwargs['encoding'] = encoding
            kwargs['labels'] = labels
            kwargs['gaussian_prior_sigma'] = gaussian_prior_sigma
            return TadmMaxentClassifier.train(train_toks, **kwargs)
        else:
            raise ValueError('Unknown algorithm %s' % algorithm)