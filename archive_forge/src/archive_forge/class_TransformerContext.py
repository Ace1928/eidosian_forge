import dataclasses
import inspect
import enum
import functools
import textwrap
from typing import (
from typing_extensions import Protocol
from cirq import circuits
@dataclasses.dataclass(frozen=True)
class TransformerContext:
    """Stores common configurable options for transformers.

    Args:
        logger: `cirq.TransformerLogger` instance, which is a stateful logger used for logging
                the actions of individual transformer stages. The same logger instance should be
                shared across different transformer calls.
        tags_to_ignore: Tuple of tags which should be ignored while applying transformations on a
                circuit. Transformers should not transform any operation marked with a tag that
                belongs to this tuple. Note that any instance of a Hashable type (like `str`,
                `cirq.VirtualTag` etc.) is a valid tag.
        deep: If true, the transformer should be recursively applied to all sub-circuits wrapped
                inside circuit operations.
    """
    logger: TransformerLogger = NoOpTransformerLogger()
    tags_to_ignore: Tuple[Hashable, ...] = ()
    deep: bool = False