import os
import re
import threading
import time
from tensorboard.backend.event_processing import data_provider
from tensorboard.backend.event_processing import plugin_event_multiplexer
from tensorboard.backend.event_processing import tag_types
from tensorboard.compat import tf
from tensorboard.data import ingester
from tensorboard.plugins.audio import metadata as audio_metadata
from tensorboard.plugins.histogram import metadata as histogram_metadata
from tensorboard.plugins.image import metadata as image_metadata
from tensorboard.plugins.pr_curve import metadata as pr_curve_metadata
from tensorboard.plugins.scalar import metadata as scalar_metadata
from tensorboard.util import tb_logging
def _get_event_file_active_filter(flags):
    """Returns a predicate for whether an event file load timestamp is active.

    Returns:
      A predicate function accepting a single UNIX timestamp float argument, or
      None if multi-file loading is not enabled.
    """
    if not flags.reload_multifile:
        return None
    inactive_secs = flags.reload_multifile_inactive_secs
    if inactive_secs == 0:
        return None
    if inactive_secs < 0:
        return lambda timestamp: True
    return lambda timestamp: timestamp + inactive_secs >= time.time()