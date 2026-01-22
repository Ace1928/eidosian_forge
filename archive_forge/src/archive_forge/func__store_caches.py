import os, sys, datetime, re
from rdflib import Graph
from ..utils import create_file_name
from . import VocabCachingInfo
import pickle
def _store_caches(self):
    """Called if the creation date, etc, have been refreshed or new, and
        all content must be put into a cache file
        """
    fname = os.path.join(self.app_data_dir, self.filename)
    try:
        _dump(self.graph, fname)
    except Exception:
        _t, value, _traceback = sys.exc_info()
        if self.report:
            self.options.add_info('Could not write cache file %s (%s)', (fname, value), VocabCachingInfo, self.uri)
    self.add_ref(self.uri, (self.filename, self.creation_date, self.expiration_date))