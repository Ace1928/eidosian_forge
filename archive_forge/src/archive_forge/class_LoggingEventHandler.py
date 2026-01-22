import os.path
import logging
import re
from .patterns import match_any_paths
from wandb_watchdog.utils import has_attribute
from wandb_watchdog.utils import unicode_paths
class LoggingEventHandler(FileSystemEventHandler):
    """Logs all the events captured."""

    def on_moved(self, event):
        super(LoggingEventHandler, self).on_moved(event)
        what = 'directory' if event.is_directory else 'file'
        logging.info('Moved %s: from %s to %s', what, event.src_path, event.dest_path)

    def on_created(self, event):
        super(LoggingEventHandler, self).on_created(event)
        what = 'directory' if event.is_directory else 'file'
        logging.info('Created %s: %s', what, event.src_path)

    def on_deleted(self, event):
        super(LoggingEventHandler, self).on_deleted(event)
        what = 'directory' if event.is_directory else 'file'
        logging.info('Deleted %s: %s', what, event.src_path)

    def on_modified(self, event):
        super(LoggingEventHandler, self).on_modified(event)
        what = 'directory' if event.is_directory else 'file'
        logging.info('Modified %s: %s', what, event.src_path)