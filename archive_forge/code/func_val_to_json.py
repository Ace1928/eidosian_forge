import logging
import os
import re
from typing import TYPE_CHECKING, Optional, Sequence, Union, cast
import wandb
from wandb import util
from .base_types.media import BatchableMedia, Media
from .base_types.wb_value import WBValue
from .image import _server_accepts_image_filenames
from .plotly import Plotly
def val_to_json(run: Optional['LocalRun'], key: str, val: 'ValToJsonType', namespace: Optional[Union[str, int]]=None, ignore_copy_err: Optional[bool]=None) -> Union[Sequence, dict]:
    if namespace is None:
        raise ValueError("val_to_json must be called with a namespace(a step number, or 'summary') argument")
    converted = val
    if isinstance(val, (int, float, str, bool)):
        return converted
    typename = util.get_full_typename(val)
    if util.is_pandas_data_frame(val):
        val = wandb.Table(dataframe=val)
    elif util.is_matplotlib_typename(typename) or util.is_plotly_typename(typename):
        val = Plotly.make_plot_media(val)
    elif isinstance(val, (list, tuple, range)) and all((isinstance(v, WBValue) for v in val)):
        assert run
        if len(val) and isinstance(val[0], BatchableMedia) and all((isinstance(v, type(val[0])) for v in val)):
            if TYPE_CHECKING:
                val = cast(Sequence['BatchableMedia'], val)
            items = _prune_max_seq(val)
            if _server_accepts_image_filenames():
                for item in items:
                    item.bind_to_run(run=run, key=key, step=namespace, ignore_copy_err=ignore_copy_err)
            else:
                for i, item in enumerate(items):
                    item.bind_to_run(run=run, key=key, step=namespace, id_=i, ignore_copy_err=ignore_copy_err)
                if run._attach_id and run._init_pid != os.getpid():
                    wandb.termwarn(f'Attempting to log a sequence of {items[0].__class__.__name__} objects from multiple processes might result in data loss. Please upgrade your wandb server', repeat=False)
            return items[0].seq_to_json(items, run, key, namespace)
        else:
            return [val_to_json(run, key, v, namespace=namespace, ignore_copy_err=ignore_copy_err) for v in val]
    if isinstance(val, WBValue):
        assert run
        if isinstance(val, Media) and (not val.is_bound()):
            if hasattr(val, '_log_type') and val._log_type in ['table', 'partitioned-table', 'joined-table']:
                sanitized_key = re.sub('[^a-zA-Z0-9_]+', '', key)
                art = wandb.Artifact(f'run-{run.id}-{sanitized_key}', 'run_table')
                art.add(val, key)
                run.log_artifact(art)
            if not (hasattr(val, '_log_type') and val._log_type in ['partitioned-table', 'joined-table']):
                val.bind_to_run(run, key, namespace)
        return val.to_json(run)
    return converted