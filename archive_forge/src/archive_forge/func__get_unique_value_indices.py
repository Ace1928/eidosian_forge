from collections import Counter, OrderedDict
from functools import partial
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import pandas.api.types
from ray.air.util.data_batch_conversion import BatchFormat
from ray.data import Dataset
from ray.data.preprocessor import Preprocessor, PreprocessorNotFittedException
from ray.util.annotations import PublicAPI
def _get_unique_value_indices(dataset: Dataset, columns: List[str], drop_na_values: bool=False, key_format: str='unique_values({0})', max_categories: Optional[Dict[str, int]]=None, encode_lists: bool=True) -> Dict[str, Dict[str, int]]:
    """If drop_na_values is True, will silently drop NA values."""
    if max_categories is None:
        max_categories = {}
    for column in max_categories:
        if column not in columns:
            raise ValueError(f'You set `max_categories` for {column}, which is not present in {columns}.')

    def get_pd_value_counts_per_column(col: pd.Series):
        if _is_series_composed_of_lists(col):
            if encode_lists:
                counter = Counter()

                def update_counter(element):
                    counter.update(element)
                    return element
                col.map(update_counter)
                return counter
            else:
                col = col.map(lambda x: tuple(x))
        return Counter(col.value_counts(dropna=False).to_dict())

    def get_pd_value_counts(df: pd.DataFrame) -> List[Dict[str, Counter]]:
        df_columns = df.columns.tolist()
        result = {}
        for col in columns:
            if col in df_columns:
                result[col] = [get_pd_value_counts_per_column(df[col])]
            else:
                raise ValueError(f"Column '{col}' does not exist in DataFrame, which has columns: {df_columns}")
        return result
    value_counts = dataset.map_batches(get_pd_value_counts, batch_format='pandas')
    final_counters = {col: Counter() for col in columns}
    for batch in value_counts.iter_batches(batch_size=None):
        for col, counters in batch.items():
            for counter in counters:
                final_counters[col] += counter
    for col in columns:
        if drop_na_values:
            counter = final_counters[col]
            counter_dict = dict(counter)
            sanitized_dict = {k: v for k, v in counter_dict.items() if not pd.isnull(k)}
            final_counters[col] = Counter(sanitized_dict)
        elif any((pd.isnull(k) for k in final_counters[col])):
            raise ValueError(f"Unable to fit column '{col}' because it contains null values. Consider imputing missing values first.")
    unique_values_with_indices = OrderedDict()
    for column in columns:
        if column in max_categories:
            unique_values_with_indices[key_format.format(column)] = {k[0]: j for j, k in enumerate(final_counters[column].most_common(max_categories[column]))}
        else:
            unique_values_with_indices[key_format.format(column)] = {k: j for j, k in enumerate(sorted(dict(final_counters[column]).keys()))}
    return unique_values_with_indices