import fnmatch
import io
import re
import tarfile
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union
from ray.data.block import BlockAccessor
from ray.data.datasource.file_based_datasource import FileBasedDatasource
from ray.util.annotations import PublicAPI
Read and decode samples from a stream.

        Note that fileselect selects files during reading, while suffixes
        selects files during the grouping step.

        Args:
            stream: File descriptor to read from.
            path: Path to the data.
            decoder: decoder or list of decoders to be applied to samples
            fileselect: Predicate for skipping files in tar decoder.
                Defaults to lambda_:False.
            suffixes: List of suffixes to be extracted. Defaults to None.
            verbose_open: Print message when opening files. Defaults to False.

        Yields:
            List[Dict[str, Any]]: List of sample (list of length 1).
        