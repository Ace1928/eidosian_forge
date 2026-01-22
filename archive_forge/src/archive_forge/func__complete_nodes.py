from typing import Callable, Optional
import numpy as np
import modin.pandas as pd
from modin.config import NPartitions
from modin.core.execution.ray.implementations.pandas_on_ray.dataframe.dataframe import (
from modin.core.storage_formats.pandas import PandasQueryCompiler
from modin.error_message import ErrorMessage
from modin.utils import get_current_execution
def _complete_nodes(self, list_of_nodes, partitions):
    """
        Run a sub-query end to end.

        Parameters
        ----------
        list_of_nodes : list of PandasQuery
            The functions that compose this query.
        partitions : list of PandasOnRayDataframeVirtualPartition
            The partitions that compose the dataframe that is input to this sub-query.

        Returns
        -------
        list of PandasOnRayDataframeVirtualPartition
            The partitions that result from computing the functions represented by `list_of_nodes`.
        """
    for node in list_of_nodes:
        if node.fan_out:
            if len(partitions) > 1:
                ErrorMessage.not_implemented('Fan out is only supported with DataFrames with 1 partition.')
            partitions[0] = partitions[0].force_materialization()
            partition_list = partitions[0].list_of_block_partitions
            partitions[0] = partitions[0].add_to_apply_calls(node.func, 0)
            partitions[0].drain_call_queue(num_splits=1)
            new_dfs = []
            for i in range(1, self.num_partitions):
                new_dfs.append(type(partitions[0])(partition_list, full_axis=partitions[0].full_axis).add_to_apply_calls(node.func, i))
                new_dfs[-1].drain_call_queue(num_splits=1)

            def reducer(df):
                df_inputs = [df]
                for df in new_dfs:
                    df_inputs.append(df.to_pandas())
                return node.reduce_fn(df_inputs)
            partitions = [partitions[0].add_to_apply_calls(reducer)]
        elif node.repartition_after:
            if len(partitions) > 1:
                ErrorMessage.not_implemented('Dynamic repartitioning is currently only supported for DataFrames with 1 partition.')
            partitions[0] = partitions[0].add_to_apply_calls(node.func).force_materialization()
            new_dfs = []

            def mask_partition(df, i):
                new_length = len(df.index) // self.num_partitions
                if i == self.num_partitions - 1:
                    return df.iloc[i * new_length:]
                return df.iloc[i * new_length:(i + 1) * new_length]
            for i in range(self.num_partitions):
                new_dfs.append(type(partitions[0])(partitions[0].list_of_block_partitions, full_axis=partitions[0].full_axis).add_to_apply_calls(mask_partition, i))
            partitions = new_dfs
        elif node.pass_partition_id:
            partitions = [part.add_to_apply_calls(node.func, i) for i, part in enumerate(partitions)]
        else:
            partitions = [part.add_to_apply_calls(node.func) for part in partitions]
    return partitions