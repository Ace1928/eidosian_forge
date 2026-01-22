import sys
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union
import wandb
def _infer_validation_row_processor(example_input: Union[Dict, Sequence], example_target: Union[Dict, Sequence, Any], class_labels_table: Optional['wandb.Table']=None, input_col_name: str='input', target_col_name: str='target') -> Callable:
    """Infers the composit processor for the validation data."""
    single_processors = {}
    if isinstance(example_input, dict):
        for key in example_input:
            key_processors = _infer_single_example_keyed_processor(example_input[key])
            for p_key in key_processors:
                single_processors[f'{key}:{p_key}'] = _bind(lambda ndx, row, key_processor, key: key_processor(ndx, row[key], None), key_processor=key_processors[p_key], key=key)
    else:
        key = input_col_name
        key_processors = _infer_single_example_keyed_processor(example_input)
        for p_key in key_processors:
            single_processors[f'{key}:{p_key}'] = _bind(lambda ndx, row, key_processor, key: key_processor(ndx, row[key], None), key_processor=key_processors[p_key], key=key)
    if isinstance(example_target, dict):
        for key in example_target:
            key_processors = _infer_single_example_keyed_processor(example_target[key], class_labels_table)
            for p_key in key_processors:
                single_processors[f'{key}:{p_key}'] = _bind(lambda ndx, row, key_processor, key: key_processor(ndx, row[key], None), key_processor=key_processors[p_key], key=key)
    else:
        key = target_col_name
        key_processors = _infer_single_example_keyed_processor(example_target, class_labels_table, example_input if not isinstance(example_input, dict) else None)
        for p_key in key_processors:
            single_processors[f'{key}:{p_key}'] = _bind(lambda ndx, row, key_processor, key: key_processor(ndx, row[key], row[input_col_name] if not isinstance(example_input, dict) else None), key_processor=key_processors[p_key], key=key)

    def processor(ndx, row):
        return {key: single_processors[key](ndx, row) for key in single_processors}
    return processor