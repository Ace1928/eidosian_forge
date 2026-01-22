from typing import Any, Callable, List, Optional, Union
from fugue import FugueWorkflow
from triad import assert_or_throw, conditional_dispatcher
from tune.concepts.dataset import TuneDataset, TuneDatasetBuilder
from tune.concepts.flow import Monitor
from tune.concepts.space import Space
from tune.concepts.logger import MetricLogger
from tune.constants import (
from tune.exceptions import TuneCompileError
from tune.iterative.objective import IterativeObjectiveFunc
from tune.noniterative.convert import to_noniterative_objective
from tune.noniterative.objective import (
from tune.noniterative.stopper import NonIterativeStopper
class TuneObjectFactory:

    def __init__(self):
        self._tmp = ''

    def set_temp_path(self, path: str) -> None:
        self._tmp = path

    def make_dataset(self, dag: FugueWorkflow, dataset: Any, df: Any=None, df_name: str=TUNE_DATASET_DF_DEFAULT_NAME, test_df: Any=None, test_df_name: str=TUNE_DATASET_VALIDATION_DF_DEFAULT_NAME, partition_keys: Optional[List[str]]=None, shuffle: bool=True, temp_path: str='') -> TuneDataset:
        assert_or_throw(dataset is not None, TuneCompileError("dataset can't be None"))
        if isinstance(dataset, TuneDataset):
            assert_or_throw(df is None, TuneCompileError("can't set df when dataset is TuneDataset"))
            return dataset
        if isinstance(dataset, Space):
            if df is None and test_df is None:
                path = ''
            else:
                path = self.get_path_or_temp(temp_path)
            builder = TuneDatasetBuilder(dataset, path)
            if df is not None:
                wdf = dag.df(df)
                if partition_keys is not None and len(partition_keys) > 0:
                    wdf = wdf.partition_by(*partition_keys)
                builder.add_df(df_name, wdf)
            if test_df is not None:
                wdf = dag.df(test_df)
                how = 'cross'
                if partition_keys is not None and len(partition_keys) > 0:
                    wdf = wdf.partition_by(*partition_keys)
                    how = 'inner'
                builder.add_df(test_df_name, wdf, how=how)
            return builder.build(dag, batch_size=1, shuffle=shuffle)
        raise TuneCompileError(f"{dataset} can't be converted to TuneDataset")

    def get_path_or_temp(self, path: str) -> str:
        if path is None or path == '':
            path = self._tmp
        assert_or_throw(path != '', TuneCompileError('path or temp path must be set'))
        return path