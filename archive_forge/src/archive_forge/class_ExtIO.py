from ._base import *
class ExtIO:

    def __init__(self, ext: str):
        self._ext = None
        self._src_ext = ext
        for e in FileExtIO:
            if ext in e.value:
                self._ext = FileExtIO[e.name]
                break

    @property
    def appendable_fs(self):
        return [FileExtIO.TXT, FileExtIO.JSONLINES, FileExtIO.TFRECORDS]

    @property
    def appendable(self):
        return self._ext in self.appendable_fs