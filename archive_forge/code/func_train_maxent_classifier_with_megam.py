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
def train_maxent_classifier_with_megam(train_toks, trace=3, encoding=None, labels=None, gaussian_prior_sigma=0, **kwargs):
    """
    Train a new ``ConditionalExponentialClassifier``, using the given
    training samples, using the external ``megam`` library.  This
    ``ConditionalExponentialClassifier`` will encode the model that
    maximizes entropy from all the models that are empirically
    consistent with ``train_toks``.

    :see: ``train_maxent_classifier()`` for parameter descriptions.
    :see: ``nltk.classify.megam``
    """
    explicit = True
    bernoulli = True
    if 'explicit' in kwargs:
        explicit = kwargs['explicit']
    if 'bernoulli' in kwargs:
        bernoulli = kwargs['bernoulli']
    if encoding is None:
        count_cutoff = kwargs.get('count_cutoff', 0)
        encoding = BinaryMaxentFeatureEncoding.train(train_toks, count_cutoff, labels=labels, alwayson_features=True)
    elif labels is not None:
        raise ValueError('Specify encoding or labels, not both')
    try:
        fd, trainfile_name = tempfile.mkstemp(prefix='nltk-')
        with open(trainfile_name, 'w') as trainfile:
            write_megam_file(train_toks, encoding, trainfile, explicit=explicit, bernoulli=bernoulli)
        os.close(fd)
    except (OSError, ValueError) as e:
        raise ValueError('Error while creating megam training file: %s' % e) from e
    options = []
    options += ['-nobias', '-repeat', '10']
    if explicit:
        options += ['-explicit']
    if not bernoulli:
        options += ['-fvals']
    if gaussian_prior_sigma:
        inv_variance = 1.0 / gaussian_prior_sigma ** 2
    else:
        inv_variance = 0
    options += ['-lambda', '%.2f' % inv_variance, '-tune']
    if trace < 3:
        options += ['-quiet']
    if 'max_iter' in kwargs:
        options += ['-maxi', '%s' % kwargs['max_iter']]
    if 'll_delta' in kwargs:
        options += ['-dpp', '%s' % abs(kwargs['ll_delta'])]
    if hasattr(encoding, 'cost'):
        options += ['-multilabel']
    options += ['multiclass', trainfile_name]
    stdout = call_megam(options)
    try:
        os.remove(trainfile_name)
    except OSError as e:
        print(f'Warning: unable to delete {trainfile_name}: {e}')
    weights = parse_megam_weights(stdout, encoding.length(), explicit)
    weights *= numpy.log2(numpy.e)
    return MaxentClassifier(encoding, weights)