import os
import shutil
import sys
from collections import ChainMap, OrderedDict, defaultdict
from typing import Any, DefaultDict, Iterable, Iterator, List, Optional, Tuple, Union
from lightning_utilities.core.apply_func import apply_to_collection
from torch import Tensor
import pytorch_lightning as pl
from lightning_fabric.utilities.data import _set_sampler_epoch
from pytorch_lightning.callbacks.progress.rich_progress import _RICH_AVAILABLE
from pytorch_lightning.loops.fetchers import _DataFetcher, _DataLoaderIterDataFetcher
from pytorch_lightning.loops.loop import _Loop
from pytorch_lightning.loops.progress import _BatchProgress
from pytorch_lightning.loops.utilities import _no_grad_context, _select_data_fetcher, _verify_dataloader_idx_requirement
from pytorch_lightning.trainer import call
from pytorch_lightning.trainer.connectors.data_connector import (
from pytorch_lightning.trainer.connectors.logger_connector.result import _OUT_DICT, _ResultCollection
from pytorch_lightning.trainer.states import RunningStage, TrainerFn
from pytorch_lightning.utilities.combined_loader import CombinedLoader
from pytorch_lightning.utilities.data import has_len_all_ranks
from pytorch_lightning.utilities.exceptions import SIGTERMException
from pytorch_lightning.utilities.model_helpers import _ModuleMode, is_overridden
from pytorch_lightning.utilities.signature_utils import is_param_in_hook_signature
@staticmethod
def _print_results(results: List[_OUT_DICT], stage: str) -> None:
    results = [{k.split('/dataloader_idx_')[0]: v for k, v in result.items()} for result in results]
    metrics_paths = {k for keys in apply_to_collection(results, dict, _EvaluationLoop._get_keys) for k in keys}
    if not metrics_paths:
        return
    metrics_strs = [':'.join(metric) for metric in metrics_paths]
    metrics_strs, metrics_paths = zip(*sorted(zip(metrics_strs, metrics_paths)))
    headers = [f'DataLoader {i}' for i in range(len(results))]
    term_size = shutil.get_terminal_size(fallback=(120, 30)).columns or 120
    max_length = int(min(max(len(max(metrics_strs, key=len)), len(max(headers, key=len)), 25), term_size / 2))
    rows: List[List[Any]] = [[] for _ in metrics_paths]
    for result in results:
        for metric, row in zip(metrics_paths, rows):
            val = _EvaluationLoop._find_value(result, metric)
            if val is not None:
                if isinstance(val, Tensor):
                    val = val.item() if val.numel() == 1 else val.tolist()
                row.append(f'{val}')
            else:
                row.append(' ')
    num_cols = int((term_size - max_length) / max_length)
    for i in range(0, len(headers), num_cols):
        table_headers = headers[i:i + num_cols]
        table_rows = [row[i:i + num_cols] for row in rows]
        table_headers.insert(0, f'{stage} Metric'.capitalize())
        if _RICH_AVAILABLE:
            from rich import get_console
            from rich.table import Column, Table
            columns = [Column(h, justify='center', style='magenta', width=max_length) for h in table_headers]
            columns[0].style = 'cyan'
            table = Table(*columns)
            for metric, row in zip(metrics_strs, table_rows):
                row.insert(0, metric)
                table.add_row(*row)
            console = get_console()
            console.print(table)
        else:
            row_format = f'{{:^{max_length}}}' * len(table_headers)
            half_term_size = int(term_size / 2)
            try:
                if sys.stdout.encoding is not None:
                    '─'.encode(sys.stdout.encoding)
            except UnicodeEncodeError:
                bar_character = '-'
            else:
                bar_character = '─'
            bar = bar_character * term_size
            lines = [bar, row_format.format(*table_headers).rstrip(), bar]
            for metric, row in zip(metrics_strs, table_rows):
                if len(metric) > half_term_size:
                    while len(metric) > half_term_size:
                        row_metric = metric[:half_term_size]
                        metric = metric[half_term_size:]
                        lines.append(row_format.format(row_metric, *row).rstrip())
                    lines.append(row_format.format(metric, ' ').rstrip())
                else:
                    lines.append(row_format.format(metric, *row).rstrip())
            lines.append(bar)
            print(os.linesep.join(lines))