import threading
from contextlib import contextmanager
from typing import Iterator, Optional
def use_const_ref_for_mutable_tensors() -> bool:
    assert _locals.use_const_ref_for_mutable_tensors is not None, 'need to initialize local.use_const_ref_for_mutable_tensors with local.parametrize'
    return _locals.use_const_ref_for_mutable_tensors