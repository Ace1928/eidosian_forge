import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
import torch
from lightning_fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_1
from lightning_fabric.utilities.load import _METADATA_FILENAME, _load_distributed_checkpoint
def _process_cli_args(args: Namespace) -> Namespace:
    if not _TORCH_GREATER_EQUAL_2_1:
        _log.error('Processing distributed checkpoints requires PyTorch >= 2.1.')
        exit(1)
    checkpoint_folder = Path(args.checkpoint_folder)
    if not checkpoint_folder.exists():
        _log.error(f'The provided checkpoint folder does not exist: {checkpoint_folder}')
        exit(1)
    if not checkpoint_folder.is_dir():
        _log.error(f'The provided checkpoint path must be a folder, containing the checkpoint shards: {checkpoint_folder}')
        exit(1)
    if not (checkpoint_folder / _METADATA_FILENAME).is_file():
        _log.error(f'Only FSDP-sharded checkpoints saved with Lightning are supported for consolidation. The provided folder is not in that format: {checkpoint_folder}')
        exit(1)
    if args.output_file is None:
        output_file = checkpoint_folder.with_suffix(checkpoint_folder.suffix + '.consolidated')
    else:
        output_file = Path(args.output_file)
    if output_file.exists():
        _log.error(f'The path for the converted checkpoint already exists. Choose a different path by providing `--output_file` or move/delete the file first: {output_file}')
        exit(1)
    return Namespace(checkpoint_folder=checkpoint_folder, output_file=output_file)