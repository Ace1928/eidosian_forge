from __future__ import absolute_import, division, print_function
from functools import partial
import numpy as np
from ..io import SEGMENT_DTYPE
from ..processors import SequentialProcessor
class CRFChordRecognitionProcessor(SequentialProcessor):
    """
    Recognise major and minor chords from learned features extracted by
    a convolutional neural network, as described in [1]_.

    Parameters
    ----------
    model : str
        File containing the CRF model. If None, use the model supplied with
        madmom.
    fps : float
        Frames per second. Must correspond to the fps of the incoming
        activations and the model.

    References
    ----------
    .. [1] Filip Korzeniowski and Gerhard Widmer,
           "A Fully Convolutional Deep Auditory Model for Musical Chord
           Recognition",
           Proceedings of IEEE International Workshop on Machine Learning for
           Signal Processing (MLSP), 2016.

    Examples
    --------
    To recognise chords using the CRFChordRecognitionProcessor, you first need
    to extract features using the CNNChordFeatureProcessor.

    >>> featproc = CNNChordFeatureProcessor()
    >>> featproc  # doctest: +ELLIPSIS
    <madmom.features.chords.CNNChordFeatureProcessor object at 0x...>

    Then, create the CRFChordRecognitionProcessor to decode a chord sequence
    from the extracted features:

    >>> decode = CRFChordRecognitionProcessor()
    >>> decode  # doctest: +ELLIPSIS
    <madmom.features.chords.CRFChordRecognitionProcessor object at 0x...>

    To transcribe the chords, you can either manually call the processors
    one after another,

    >>> feats = featproc('tests/data/audio/sample2.wav')
    >>> decode(feats)
    ... # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS +IGNORE_UNICODE
    ... # doctest: +NORMALIZE_ARRAYS
    array([(0. , 0.2, 'N'), (0.2, 1.6, 'F:maj'),
           (1.6, 2.4..., 'A:maj'), (2.4..., 4.1, 'D:min')],
          dtype=[('start', '<f8'), ('end', '<f8'), ('label', 'O')])

    or create a `madmom.processors.SequentialProcessor` that connects them:

    >>> from madmom.processors import SequentialProcessor
    >>> chordrec = SequentialProcessor([featproc, decode])
    >>> chordrec('tests/data/audio/sample2.wav')
    ... # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS +IGNORE_UNICODE
    ... # doctest: +NORMALIZE_ARRAYS
    array([(0. , 0.2, 'N'), (0.2, 1.6, 'F:maj'),
           (1.6, 2.4..., 'A:maj'), (2.4..., 4.1, 'D:min')],
          dtype=[('start', '<f8'), ('end', '<f8'), ('label', 'O')])
    """

    def __init__(self, model=None, fps=10, **kwargs):
        from ..ml.crf import ConditionalRandomField
        from ..models import CHORDS_CFCRF
        crf = ConditionalRandomField.load(model or CHORDS_CFCRF[0])
        lbl = partial(majmin_targets_to_chord_labels, fps=fps)
        super(CRFChordRecognitionProcessor, self).__init__((crf, lbl))