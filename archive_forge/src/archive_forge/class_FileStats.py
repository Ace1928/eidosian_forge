import threading
from typing import MutableMapping, NamedTuple
import wandb
class FileStats(NamedTuple):
    deduped: bool
    total: int
    uploaded: int
    failed: bool
    artifact_file: bool