import collections
import dataclasses
import threading
from typing import Optional
from tensorboard.backend.event_processing import directory_loader
from tensorboard.backend.event_processing import directory_watcher
from tensorboard.backend.event_processing import event_file_loader
from tensorboard.backend.event_processing import event_util
from tensorboard.backend.event_processing import io_wrapper
from tensorboard.backend.event_processing import plugin_asset_util
from tensorboard.backend.event_processing import reservoir
from tensorboard.backend.event_processing import tag_types
from tensorboard.compat.proto import config_pb2
from tensorboard.compat.proto import event_pb2
from tensorboard.compat.proto import graph_pb2
from tensorboard.compat.proto import meta_graph_pb2
from tensorboard.compat.proto import tensor_pb2
from tensorboard.util import tb_logging
def _GeneratorFromPath(path, event_file_active_filter=None, detect_file_replacement=None):
    """Create an event generator for file or directory at given path string."""
    if not path:
        raise ValueError('path must be a valid string')
    if io_wrapper.IsSummaryEventsFile(path):
        return event_file_loader.EventFileLoader(path, detect_file_replacement)
    elif event_file_active_filter:
        loader_factory = lambda path: event_file_loader.TimestampedEventFileLoader(path, detect_file_replacement)
        return directory_loader.DirectoryLoader(path, loader_factory, path_filter=io_wrapper.IsSummaryEventsFile, active_filter=event_file_active_filter)
    else:
        loader_factory = lambda path: event_file_loader.EventFileLoader(path, detect_file_replacement)
        return directory_watcher.DirectoryWatcher(path, loader_factory, io_wrapper.IsSummaryEventsFile)