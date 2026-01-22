import sys
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union
import wandb
def _infer_single_example_keyed_processor(example: Union[Sequence, Any], class_labels_table: Optional['wandb.Table']=None, possible_base_example: Optional[Union[Sequence, Any]]=None) -> Dict[str, Callable]:
    """Infers a processor from a single example.

    Infers a processor from a single example with optional class_labels_table
    and base_example. Base example is useful for cases such as segmentation masks
    """
    shape = _get_example_shape(example)
    processors: Dict[str, Callable] = {}
    if class_labels_table is not None and len(shape) == 1 and (shape[0] == len(class_labels_table.data)):
        np = wandb.util.get_module('numpy', required='Infering processors require numpy')
        class_names = class_labels_table.get_column('label')
        processors['max_class'] = lambda n, d, p: class_labels_table.index_ref(np.argmax(d))
        values = np.unique(example)
        is_one_hot = len(values) == 2 and set(values) == {0, 1}
        if not is_one_hot:
            processors['score'] = lambda n, d, p: {class_names[i]: d[i] for i in range(shape[0])}
    elif len(shape) == 1 and shape[0] == 1 and (isinstance(example[0], int) or (hasattr(example, 'tolist') and isinstance(example.tolist()[0], int))):
        if class_labels_table is not None:
            processors['class'] = lambda n, d, p: class_labels_table.index_ref(d[0]) if d[0] < len(class_labels_table.data) else d[0]
        else:
            processors['val'] = lambda n, d, p: d[0]
    elif len(shape) == 1:
        np = wandb.util.get_module('numpy', required='Infering processors require numpy')
        if shape[0] <= 10:
            processors['node'] = lambda n, d, p: [d[i].tolist() if hasattr(d[i], 'tolist') else d[i] for i in range(shape[0])]
        processors['argmax'] = lambda n, d, p: np.argmax(d)
        values = np.unique(example)
        is_one_hot = len(values) == 2 and set(values) == {0, 1}
        if not is_one_hot:
            processors['argmin'] = lambda n, d, p: np.argmin(d)
    elif len(shape) == 2 and CAN_INFER_IMAGE_AND_VIDEO:
        if class_labels_table is not None and possible_base_example is not None and (shape == _get_example_shape(possible_base_example)):
            processors['image'] = lambda n, d, p: wandb.Image(p, masks={'masks': {'mask_data': d, 'class_labels': class_labels_table.get_column('label')}})
        else:
            processors['image'] = lambda n, d, p: wandb.Image(d)
    elif len(shape) == 3 and CAN_INFER_IMAGE_AND_VIDEO:
        processors['image'] = lambda n, d, p: wandb.Image(d)
    elif len(shape) == 4 and CAN_INFER_IMAGE_AND_VIDEO:
        processors['video'] = lambda n, d, p: wandb.Video(d)
    return processors