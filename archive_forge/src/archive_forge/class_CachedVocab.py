import os, sys, datetime, re
from rdflib import Graph
from ..utils import create_file_name
from . import VocabCachingInfo
import pickle
class CachedVocab(CachedVocabIndex):
    """
    Cache for a specific vocab. The content of the cache is the graph. These are also the data that are stored
    on the disc (in pickled form)

    @ivar graph: the RDF graph
    @ivar URI: vocabulary URI
    @ivar filename: file name (not the complete path) of the cached version
    @ivar creation_date: creation date of the cache
    @type creation_date: datetime
    @ivar expiration_date: expiration date of the cache
    @type expiration_date: datetime
    @cvar runtime_cache : a run time cache for already 'seen' vocabulary files. Apart from (marginally) speeding up processing, this also prevents recursion
    @type runtime_cache : dictionary
    """

    def __init__(self, URI, options=None, verify=True):
        """
        @param URI: real URI for the vocabulary file
        @param options: the error handler (option) object to send warnings to
        @type options: L{options.Options}
        @param verify: whether the SSL certificate needs to be verified.
        @type verify: bool
        """
        self.uri = URI
        self.filename, self.creation_date, self.expiration_date = ('', None, None)
        self.graph = Graph()
        try:
            CachedVocabIndex.__init__(self, options)
            vocab_reference = self.get_ref(URI)
            self.caching = True
        except Exception:
            _t, value, _traceback = sys.exc_info()
            if self.report:
                options.add_info('Could not access the vocabulary cache area %s' % value, VocabCachingInfo, URI)
            vocab_reference = None
            self.caching = False
        if vocab_reference == None:
            if self.report:
                options.add_info('No cache exists for %s, generating one' % URI, VocabCachingInfo)
            if self._get_vocab_data(verify, newCache=True) and self.caching:
                self.filename = create_file_name(self.uri)
                self._store_caches()
                if self.report:
                    options.add_info('Generated a cache for %s, with an expiration date of %s' % (URI, self.expiration_date), VocabCachingInfo, URI)
        else:
            self.filename, self.creation_date, self.expiration_date = vocab_reference
            if self.report:
                options.add_info('Found a cache for %s, expiring on %s' % (URI, self.expiration_date), VocabCachingInfo)
            if options.refresh_vocab_cache == False and datetime.datetime.utcnow() <= self.expiration_date:
                if self.report:
                    options.add_info('Cache for %s is still valid; extracting the data' % URI, VocabCachingInfo)
                fname = os.path.join(self.app_data_dir, self.filename)
                try:
                    self.graph = _load(fname)
                except Exception:
                    t, value, traceback = sys.exc_info()
                    sys.excepthook(t, value, traceback)
                    if self.report:
                        options.add_info('Could not access the vocab cache %s (%s)' % (value, fname), VocabCachingInfo, URI)
            else:
                if self.report:
                    if options.refresh_vocab_cache == True:
                        options.add_info('Time check is bypassed; refreshing the cache for %s' % URI, VocabCachingInfo)
                    else:
                        options.add_info('Cache timeout; refreshing the cache for %s' % URI, VocabCachingInfo)
                if self._get_vocab_data(verify, newCache=False) == False:
                    if self.report:
                        options.add_info('Could not refresh vocabulary cache for %s, using the old cache, extended its expiration time by an hour (network problems?)' % URI, VocabCachingInfo, URI)
                    fname = os.path.join(self.app_data_dir, self.filename)
                    try:
                        self.graph = _load(fname)
                        self.expiration_date = datetime.datetime.utcnow() + datetime.timedelta(hours=1)
                    except Exception:
                        t, value, traceback = sys.exc_info()
                        sys.excepthook(t, value, traceback)
                        if self.report:
                            options.add_info('Could not access the vocabulary cache %s (%s)' % (value, fname), VocabCachingInfo, URI)
                self.creation_date = datetime.datetime.utcnow()
                if self.report:
                    options.add_info('Generated a new cache for %s, with an expiration date of %s' % (URI, self.expiration_date), VocabCachingInfo, URI)
                self._store_caches()

    def _get_vocab_data(self, verify, newCache=True):
        """Just a macro like function to get the data to be cached"""
        from .process import return_graph
        self.graph, self.expiration_date = return_graph(self.uri, self.options, newCache, verify)
        return self.graph != None

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