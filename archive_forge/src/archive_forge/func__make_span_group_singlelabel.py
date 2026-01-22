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
def _make_span_group_singlelabel(self, doc: Doc, indices: Ints2d, scores: Floats2d, allow_overlap: bool=True) -> SpanGroup:
    """Find the argmax label for each span."""
    if scores.size == 0:
        return SpanGroup(doc, name=self.key)
    scores = self.model.ops.to_numpy(scores)
    indices = self.model.ops.to_numpy(indices)
    predicted = scores.argmax(axis=1)
    argmax_scores = numpy.take_along_axis(scores, numpy.expand_dims(predicted, 1), axis=1)
    keeps = numpy.ones(predicted.shape, dtype=bool)
    if self.add_negative_label:
        keeps = numpy.logical_and(keeps, predicted != self._negative_label_i)
    threshold = self.cfg['threshold']
    if threshold is not None:
        keeps = numpy.logical_and(keeps, (argmax_scores >= threshold).squeeze())
    if not allow_overlap:
        sort_idx = (argmax_scores.squeeze() * -1).argsort()
        argmax_scores = argmax_scores[sort_idx]
        predicted = predicted[sort_idx]
        indices = indices[sort_idx]
        keeps = keeps[sort_idx]
    seen = _Intervals()
    spans = SpanGroup(doc, name=self.key)
    attrs_scores = []
    for i in range(indices.shape[0]):
        if not keeps[i]:
            continue
        label = predicted[i]
        start = indices[i, 0]
        end = indices[i, 1]
        if not allow_overlap:
            if (start, end) in seen:
                continue
            else:
                seen.add(start, end)
        attrs_scores.append(argmax_scores[i])
        spans.append(Span(doc, start, end, label=self.labels[label]))
    spans.attrs['scores'] = numpy.array(attrs_scores)
    return spans