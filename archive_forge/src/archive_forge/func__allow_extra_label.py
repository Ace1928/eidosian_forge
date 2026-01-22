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
def _allow_extra_label(self) -> None:
    """Raise an error if the component can not add any more labels."""
    nO = None
    if self.model.has_dim('nO'):
        nO = self.model.get_dim('nO')
    elif self.model.has_ref('output_layer') and self.model.get_ref('output_layer').has_dim('nO'):
        nO = self.model.get_ref('output_layer').get_dim('nO')
    if nO is not None and nO == self._n_labels:
        if not self.is_resizable:
            raise ValueError(Errors.E922.format(name=self.name, nO=self.model.get_dim('nO')))