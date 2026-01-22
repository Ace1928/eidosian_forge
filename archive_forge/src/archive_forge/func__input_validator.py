from typing import Dict, Literal, Sequence, Tuple, Union
from torch import Tensor
def _input_validator(preds: Sequence[Dict[str, Tensor]], targets: Sequence[Dict[str, Tensor]], iou_type: Union[Literal['bbox', 'segm'], Tuple[Literal['bbox', 'segm']]]='bbox', ignore_score: bool=False) -> None:
    """Ensure the correct input format of `preds` and `targets`."""
    if isinstance(iou_type, str):
        iou_type = (iou_type,)
    name_map = {'bbox': 'boxes', 'segm': 'masks'}
    if any((tp not in name_map for tp in iou_type)):
        raise Exception(f'IOU type {iou_type} is not supported')
    item_val_name = [name_map[tp] for tp in iou_type]
    if not isinstance(preds, Sequence):
        raise ValueError(f'Expected argument `preds` to be of type Sequence, but got {preds}')
    if not isinstance(targets, Sequence):
        raise ValueError(f'Expected argument `target` to be of type Sequence, but got {targets}')
    if len(preds) != len(targets):
        raise ValueError(f'Expected argument `preds` and `target` to have the same length, but got {len(preds)} and {len(targets)}')
    for k in [*item_val_name, 'labels'] + (['scores'] if not ignore_score else []):
        if any((k not in p for p in preds)):
            raise ValueError(f'Expected all dicts in `preds` to contain the `{k}` key')
    for k in [*item_val_name, 'labels']:
        if any((k not in p for p in targets)):
            raise ValueError(f'Expected all dicts in `target` to contain the `{k}` key')
    for ivn in item_val_name:
        if not all((isinstance(pred[ivn], Tensor) for pred in preds)):
            raise ValueError(f'Expected all {ivn} in `preds` to be of type Tensor')
    if not ignore_score and (not all((isinstance(pred['scores'], Tensor) for pred in preds))):
        raise ValueError('Expected all scores in `preds` to be of type Tensor')
    if not all((isinstance(pred['labels'], Tensor) for pred in preds)):
        raise ValueError('Expected all labels in `preds` to be of type Tensor')
    for ivn in item_val_name:
        if not all((isinstance(target[ivn], Tensor) for target in targets)):
            raise ValueError(f'Expected all {ivn} in `target` to be of type Tensor')
    if not all((isinstance(target['labels'], Tensor) for target in targets)):
        raise ValueError('Expected all labels in `target` to be of type Tensor')
    for i, item in enumerate(targets):
        for ivn in item_val_name:
            if item[ivn].size(0) != item['labels'].size(0):
                raise ValueError(f"Input '{ivn}' and labels of sample {i} in targets have a different length (expected {item[ivn].size(0)} labels, got {item['labels'].size(0)})")
    if ignore_score:
        return
    for i, item in enumerate(preds):
        for ivn in item_val_name:
            if not item[ivn].size(0) == item['labels'].size(0) == item['scores'].size(0):
                raise ValueError(f"Input '{ivn}', labels and scores of sample {i} in predictions have a different length (expected {item[ivn].size(0)} labels and scores, got {item['labels'].size(0)} labels and {item['scores'].size(0)})")