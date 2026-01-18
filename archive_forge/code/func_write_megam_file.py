import subprocess
from nltk.internals import find_binary
def write_megam_file(train_toks, encoding, stream, bernoulli=True, explicit=True):
    """
    Generate an input file for ``megam`` based on the given corpus of
    classified tokens.

    :type train_toks: list(tuple(dict, str))
    :param train_toks: Training data, represented as a list of
        pairs, the first member of which is a feature dictionary,
        and the second of which is a classification label.

    :type encoding: MaxentFeatureEncodingI
    :param encoding: A feature encoding, used to convert featuresets
        into feature vectors. May optionally implement a cost() method
        in order to assign different costs to different class predictions.

    :type stream: stream
    :param stream: The stream to which the megam input file should be
        written.

    :param bernoulli: If true, then use the 'bernoulli' format.  I.e.,
        all joint features have binary values, and are listed iff they
        are true.  Otherwise, list feature values explicitly.  If
        ``bernoulli=False``, then you must call ``megam`` with the
        ``-fvals`` option.

    :param explicit: If true, then use the 'explicit' format.  I.e.,
        list the features that would fire for any of the possible
        labels, for each token.  If ``explicit=True``, then you must
        call ``megam`` with the ``-explicit`` option.
    """
    labels = encoding.labels()
    labelnum = {label: i for i, label in enumerate(labels)}
    for featureset, label in train_toks:
        if hasattr(encoding, 'cost'):
            stream.write(':'.join((str(encoding.cost(featureset, label, l)) for l in labels)))
        else:
            stream.write('%d' % labelnum[label])
        if not explicit:
            _write_megam_features(encoding.encode(featureset, label), stream, bernoulli)
        else:
            for l in labels:
                stream.write(' #')
                _write_megam_features(encoding.encode(featureset, l), stream, bernoulli)
        stream.write('\n')