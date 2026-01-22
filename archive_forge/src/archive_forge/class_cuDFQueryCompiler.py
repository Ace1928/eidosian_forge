import numpy as np
import pandas
from modin.core.storage_formats.pandas.query_compiler import PandasQueryCompiler
class cuDFQueryCompiler(PandasQueryCompiler):

    def transpose(self, *args, **kwargs):
        """Transposes this QueryCompiler.

        Returns:
            Transposed new QueryCompiler.
        """
        if len(np.unique(self._modin_frame.dtypes.values)) != 1:
            return self.default_to_pandas(pandas.DataFrame.transpose)
        return self.__constructor__(self._modin_frame.transpose())

    def write_items(self, row_numeric_index, col_numeric_index, item, need_columns_reindex=True):
        from modin.pandas.utils import broadcast_item, is_scalar

        def iloc_mut(partition, row_internal_indices, col_internal_indices, item):
            partition = partition.copy()
            unique_items = np.unique(item)
            if (row_internal_indices == col_internal_indices).all() and len(unique_items) == 1:
                partition.iloc[row_internal_indices] = unique_items[0]
            else:
                permutations_col = np.vstack([col_internal_indices] * len(col_internal_indices)).T.flatten()
                permutations_row = np.hstack(row_internal_indices * len(row_internal_indices))
                for i, j, it in zip(permutations_row, permutations_col, item.flatten()):
                    partition.iloc[i, j] = it
            return partition
        if not is_scalar(item):
            broadcasted_item, _ = broadcast_item(self, row_numeric_index, col_numeric_index, item, need_columns_reindex=need_columns_reindex)
        else:
            broadcasted_item = item
        new_modin_frame = self._modin_frame.apply_select_indices(axis=None, func=iloc_mut, row_labels=row_numeric_index, col_labels=col_numeric_index, new_index=self.index, new_columns=self.columns, keep_remaining=True, item_to_distribute=broadcasted_item)
        return self.__constructor__(new_modin_frame)