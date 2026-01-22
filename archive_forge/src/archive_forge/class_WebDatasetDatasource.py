import fnmatch
import io
import re
import tarfile
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union
from ray.data.block import BlockAccessor
from ray.data.datasource.file_based_datasource import FileBasedDatasource
from ray.util.annotations import PublicAPI
@PublicAPI(stability='alpha')
class WebDatasetDatasource(FileBasedDatasource):
    """A Datasource for WebDataset datasets (tar format with naming conventions)."""
    _FILE_EXTENSIONS = ['tar']

    def __init__(self, paths: Union[str, List[str]], decoder: Optional[Union[bool, str, callable, list]]=True, fileselect: Optional[Union[bool, callable, list]]=None, filerename: Optional[Union[bool, callable, list]]=None, suffixes: Optional[Union[bool, callable, list]]=None, verbose_open: bool=False, **file_based_datasource_kwargs):
        super().__init__(paths, **file_based_datasource_kwargs)
        self.decoder = decoder
        self.fileselect = fileselect
        self.filerename = filerename
        self.suffixes = suffixes
        self.verbose_open = verbose_open

    def _read_stream(self, stream: 'pyarrow.NativeFile', path: str):
        """Read and decode samples from a stream.

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
        """
        import pandas as pd
        files = _tar_file_iterator(stream, fileselect=self.fileselect, filerename=self.filerename, verbose_open=self.verbose_open)
        samples = _group_by_keys(files, meta=dict(__url__=path), suffixes=self.suffixes)
        for sample in samples:
            if self.decoder is not None:
                sample = _apply_list(self.decoder, sample, default=_default_decoder)
            yield pd.DataFrame({k: [v] for k, v in sample.items()})