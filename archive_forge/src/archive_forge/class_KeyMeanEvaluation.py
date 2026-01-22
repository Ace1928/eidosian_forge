from collections import Counter
from . import EvaluationMixin, evaluation_io
from ..io import load_key
class KeyMeanEvaluation(EvaluationMixin):
    """
    Class for averaging key evaluations.

    Parameters
    ----------
    eval_objects : list
        Key evaluation objects.
    name : str, optional
        Name to be displayed.

    """
    METRIC_NAMES = [('correct', 'Correct'), ('fifth', 'Fifth'), ('relative', 'Relative'), ('parallel', 'Parallel'), ('other', 'Other'), ('weighted', 'Weighted')]

    def __init__(self, eval_objects, name=None):
        self.name = name or 'mean for {:d} files'.format(len(eval_objects))
        n = len(eval_objects)
        c = Counter((e.error_category for e in eval_objects))
        self.correct = float(c['correct']) / n
        self.fifth = float(c['fifth']) / n
        self.relative = float(c['relative']) / n
        self.parallel = float(c['parallel']) / n
        self.other = float(c['other']) / n
        self.weighted = sum((e.score for e in eval_objects)) / n

    def tostring(self, **kwargs):
        return '{}\n  Weighted: {:.3f}  Correct: {:.3f}  Fifth: {:.3f}  Relative: {:.3f}  Parallel: {:.3f}  Other: {:.3f}'.format(self.name, self.weighted, self.correct, self.fifth, self.relative, self.parallel, self.other)