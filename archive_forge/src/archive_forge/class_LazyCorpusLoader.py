import gc
import re
import nltk
class LazyCorpusLoader:
    """
    To see the API documentation for this lazily loaded corpus, first
    run corpus.ensure_loaded(), and then run help(this_corpus).

    LazyCorpusLoader is a proxy object which is used to stand in for a
    corpus object before the corpus is loaded.  This allows NLTK to
    create an object for each corpus, but defer the costs associated
    with loading those corpora until the first time that they're
    actually accessed.

    The first time this object is accessed in any way, it will load
    the corresponding corpus, and transform itself into that corpus
    (by modifying its own ``__class__`` and ``__dict__`` attributes).

    If the corpus can not be found, then accessing this object will
    raise an exception, displaying installation instructions for the
    NLTK data package.  Once they've properly installed the data
    package (or modified ``nltk.data.path`` to point to its location),
    they can then use the corpus object without restarting python.

    :param name: The name of the corpus
    :type name: str
    :param reader_cls: The specific CorpusReader class, e.g. PlaintextCorpusReader, WordListCorpusReader
    :type reader: nltk.corpus.reader.api.CorpusReader
    :param nltk_data_subdir: The subdirectory where the corpus is stored.
    :type nltk_data_subdir: str
    :param `*args`: Any other non-keywords arguments that `reader_cls` might need.
    :param `**kwargs`: Any other keywords arguments that `reader_cls` might need.
    """

    def __init__(self, name, reader_cls, *args, **kwargs):
        from nltk.corpus.reader.api import CorpusReader
        assert issubclass(reader_cls, CorpusReader)
        self.__name = self.__name__ = name
        self.__reader_cls = reader_cls
        if 'nltk_data_subdir' in kwargs:
            self.subdir = kwargs['nltk_data_subdir']
            kwargs.pop('nltk_data_subdir', None)
        else:
            self.subdir = 'corpora'
        self.__args = args
        self.__kwargs = kwargs

    def __load(self):
        zip_name = re.sub('(([^/]+)(/.*)?)', '\\2.zip/\\1/', self.__name)
        if TRY_ZIPFILE_FIRST:
            try:
                root = nltk.data.find(f'{self.subdir}/{zip_name}')
            except LookupError as e:
                try:
                    root = nltk.data.find(f'{self.subdir}/{self.__name}')
                except LookupError:
                    raise e
        else:
            try:
                root = nltk.data.find(f'{self.subdir}/{self.__name}')
            except LookupError as e:
                try:
                    root = nltk.data.find(f'{self.subdir}/{zip_name}')
                except LookupError:
                    raise e
        corpus = self.__reader_cls(root, *self.__args, **self.__kwargs)
        args, kwargs = (self.__args, self.__kwargs)
        name, reader_cls = (self.__name, self.__reader_cls)
        self.__dict__ = corpus.__dict__
        self.__class__ = corpus.__class__

        def _unload(self):
            lazy_reader = LazyCorpusLoader(name, reader_cls, *args, **kwargs)
            self.__dict__ = lazy_reader.__dict__
            self.__class__ = lazy_reader.__class__
            gc.collect()
        self._unload = _make_bound_method(_unload, self)

    def __getattr__(self, attr):
        if attr == '__bases__':
            raise AttributeError("LazyCorpusLoader object has no attribute '__bases__'")
        self.__load()
        return getattr(self, attr)

    def __repr__(self):
        return '<{} in {!r} (not loaded yet)>'.format(self.__reader_cls.__name__, '.../corpora/' + self.__name)

    def _unload(self):
        pass