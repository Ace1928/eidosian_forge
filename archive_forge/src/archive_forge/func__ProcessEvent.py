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
def _ProcessEvent(self, event):
    """Called whenever an event is loaded."""
    if self._first_event_timestamp is None:
        self._first_event_timestamp = event.wall_time
    if event.HasField('source_metadata'):
        new_source_writer = event_util.GetSourceWriter(event.source_metadata)
        if self._source_writer and self._source_writer != new_source_writer:
            logger.info('Found new source writer for event.proto. Old: {0}, New: {1}'.format(self._source_writer, new_source_writer))
        self._source_writer = new_source_writer
    if event.HasField('file_version'):
        new_file_version = event_util.ParseFileVersion(event.file_version)
        if self.file_version and self.file_version != new_file_version:
            logger.warning('Found new file_version for event.proto. This will affect purging logic for TensorFlow restarts. Old: {0} New: {1}'.format(self.file_version, new_file_version))
        self.file_version = new_file_version
    self._MaybePurgeOrphanedData(event)
    if event.HasField('graph_def'):
        if self._graph is not None:
            logger.warning('Found more than one graph event per run, or there was a metagraph containing a graph_def, as well as one or more graph events.  Overwriting the graph with the newest event.')
        self._graph = event.graph_def
        self._graph_from_metagraph = False
    elif event.HasField('meta_graph_def'):
        if self._meta_graph is not None:
            logger.warning('Found more than one metagraph event per run. Overwriting the metagraph with the newest event.')
        self._meta_graph = event.meta_graph_def
        if self._graph is None or self._graph_from_metagraph:
            meta_graph = meta_graph_pb2.MetaGraphDef()
            meta_graph.ParseFromString(self._meta_graph)
            if meta_graph.graph_def:
                if self._graph is not None:
                    logger.warning('Found multiple metagraphs containing graph_defs,but did not find any graph events.  Overwriting the graph with the newest metagraph version.')
                self._graph_from_metagraph = True
                self._graph = meta_graph.graph_def.SerializeToString()
    elif event.HasField('tagged_run_metadata'):
        tag = event.tagged_run_metadata.tag
        if tag in self._tagged_metadata:
            logger.warning('Found more than one "run metadata" event with tag ' + tag + '. Overwriting it with the newest event.')
        self._tagged_metadata[tag] = event.tagged_run_metadata.run_metadata
    elif event.HasField('summary'):
        for value in event.summary.value:
            if value.HasField('metadata'):
                tag = value.tag
                if tag not in self.summary_metadata:
                    self.summary_metadata[tag] = value.metadata
                    plugin_data = value.metadata.plugin_data
                    if plugin_data.plugin_name:
                        with self._plugin_tag_lock:
                            self._plugin_to_tag_to_content[plugin_data.plugin_name][tag] = plugin_data.content
                    else:
                        logger.warning('This summary with tag %r is oddly not associated with a plugin.', tag)
            if value.HasField('tensor'):
                datum = value.tensor
                tag = value.tag
                if not tag:
                    tag = value.node_name
                self._ProcessTensor(tag, event.wall_time, event.step, datum)