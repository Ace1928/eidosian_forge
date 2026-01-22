from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union, cast
import numpy
from thinc.api import Config, Model, Ops, Optimizer, get_current_ops, set_dropout_rate
from thinc.types import Floats2d, Ints1d, Ints2d, Ragged
from ..compat import Protocol, runtime_checkable
from ..errors import Errors
from ..language import Language
from ..scorer import Scorer
from ..tokens import Doc, Span, SpanGroup
from ..training import Example, validate_examples
from ..util import registry
from ..vocab import Vocab
from .trainable_pipe import TrainablePipe
def _make_span_group_multilabel(self, doc: Doc, indices: Ints2d, scores: Floats2d) -> SpanGroup:
    """Find the top-k labels for each span (k=max_positive)."""
    spans = SpanGroup(doc, name=self.key)
    if scores.size == 0:
        return spans
    scores = self.model.ops.to_numpy(scores)
    indices = self.model.ops.to_numpy(indices)
    threshold = self.cfg['threshold']
    max_positive = self.cfg['max_positive']
    keeps = scores >= threshold
    if max_positive is not None:
        assert isinstance(max_positive, int)
        if self.add_negative_label:
            negative_scores = numpy.copy(scores[:, self._negative_label_i])
            scores[:, self._negative_label_i] = -numpy.inf
            ranked = (scores * -1).argsort()
            scores[:, self._negative_label_i] = negative_scores
        else:
            ranked = (scores * -1).argsort()
        span_filter = ranked[:, max_positive:]
        for i, row in enumerate(span_filter):
            keeps[i, row] = False
    attrs_scores = []
    for i in range(indices.shape[0]):
        start = indices[i, 0]
        end = indices[i, 1]
        for j, keep in enumerate(keeps[i]):
            if keep:
                if j != self._negative_label_i:
                    spans.append(Span(doc, start, end, label=self.labels[j]))
                    attrs_scores.append(scores[i, j])
    spans.attrs['scores'] = numpy.array(attrs_scores)
    return spans