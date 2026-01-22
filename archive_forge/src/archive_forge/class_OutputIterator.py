from typing import Iterable, Iterator, Optional
from .execution_options import ExecutionOptions
from .physical_operator import PhysicalOperator
from .ref_bundle import RefBundle
from ray.data._internal.stats import DatasetStats
class OutputIterator(Iterator[RefBundle]):
    """Iterator used to access the output of an Executor execution.

    This is a blocking iterator. Datasets guarantees that all its iterators are
    thread-safe (i.e., multiple threads can block on them at the same time).
    """

    def __init__(self, base: Iterable[RefBundle]):
        self._it = iter(base)

    def get_next(self, output_split_idx: Optional[int]=None) -> RefBundle:
        """Can be used to pull outputs by a specified output index.

        This is used to support the streaming_split() API, where the output of a
        streaming execution is to be consumed by multiple processes.

        Args:
            output_split_idx: The output split index to get results for. This arg is
                only allowed for iterators created by `Dataset.streaming_split()`.

        Raises:
            StopIteration if there are no more outputs to return.
        """
        if output_split_idx is not None:
            raise NotImplementedError()
        return next(self._it)

    def __next__(self) -> RefBundle:
        return self.get_next()