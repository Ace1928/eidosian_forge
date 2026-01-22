import logging
from typing import Any, Dict, List, Sequence
import wandb
from wandb.sdk.integration_utils.auto_logging import Response
from .utils import (
def add_data_to_table(self, image: Any, loggable_kwarg_chunks: List, idx: int) -> None:
    """Populate the row of the `wandb.Table`.

        Arguments:
            image: (Any) The generated images, audio, video, etc. from the Diffusion
                Pipeline's response.
            loggable_kwarg_chunks: (List) Loggable chunks of kwargs.
            idx: (int) Chunk index.
        """
    table_row = []
    kwarg_actions = SUPPORTED_MULTIMODAL_PIPELINES[self.pipeline_name]['kwarg-actions']
    for column_idx, loggable_kwarg_chunk in enumerate(loggable_kwarg_chunks):
        if kwarg_actions[column_idx] is None:
            table_row.append(loggable_kwarg_chunk[idx] if loggable_kwarg_chunk[idx] is not None else '')
        else:
            table_row.append(kwarg_actions[column_idx](loggable_kwarg_chunk[idx]))
    if 'output-type' not in SUPPORTED_MULTIMODAL_PIPELINES[self.pipeline_name]:
        table_row.append(wandb.Image(image))
    elif SUPPORTED_MULTIMODAL_PIPELINES[self.pipeline_name]['output-type'] == 'video':
        table_row.append(wandb.Video(postprocess_pils_to_np(image), fps=4))
    elif SUPPORTED_MULTIMODAL_PIPELINES[self.pipeline_name]['output-type'] == 'audio':
        table_row.append(wandb.Audio(image, sample_rate=16000))
    self.wandb_table.add_data(*table_row)