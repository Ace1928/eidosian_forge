import gc
import re
import nltk
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