import contextlib
import dataclasses
import shutil
import tempfile
from pathlib import Path
from typing import Generator, Generic, Iterable, List, Optional, TypeVar, Union
import catalogue
import confection
@my_registry.optimizers('Adam.v1')
def Adam(learn_rate: FloatOrSeq=0.001, *, beta1: FloatOrSeq=0.001, beta2: FloatOrSeq=0.001, use_averages: bool=True):
    """
    Mocks optimizer generation. Note that the returned object is not actually an optimizer. This function is merely used
    to illustrate how to use the function registry, e.g. with thinc.
    """

    @dataclasses.dataclass
    class Optimizer:
        learn_rate: FloatOrSeq
        beta1: FloatOrSeq
        beta2: FloatOrSeq
        use_averages: bool
    return Optimizer(learn_rate=learn_rate, beta1=beta1, beta2=beta2, use_averages=use_averages)