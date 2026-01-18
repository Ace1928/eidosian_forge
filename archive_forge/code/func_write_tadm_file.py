import subprocess
import sys
from nltk.internals import find_binary
def write_tadm_file(train_toks, encoding, stream):
    """
    Generate an input file for ``tadm`` based on the given corpus of
    classified tokens.

    :type train_toks: list(tuple(dict, str))
    :param train_toks: Training data, represented as a list of
        pairs, the first member of which is a feature dictionary,
        and the second of which is a classification label.
    :type encoding: TadmEventMaxentFeatureEncoding
    :param encoding: A feature encoding, used to convert featuresets
        into feature vectors.
    :type stream: stream
    :param stream: The stream to which the ``tadm`` input file should be
        written.
    """
    labels = encoding.labels()
    for featureset, label in train_toks:
        length_line = '%d\n' % len(labels)
        stream.write(length_line)
        for known_label in labels:
            v = encoding.encode(featureset, known_label)
            line = '%d %d %s\n' % (int(label == known_label), len(v), ' '.join(('%d %d' % u for u in v)))
            stream.write(line)