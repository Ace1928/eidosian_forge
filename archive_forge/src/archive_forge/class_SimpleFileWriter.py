import base64
import io
import re
import requests
import fsspec
class SimpleFileWriter(fsspec.spec.AbstractBufferedFile):

    def _upload_chunk(self, final=False):
        """Never uploads a chunk until file is done

        Not suitable for large files
        """
        if final is False:
            return False
        self.buffer.seek(0)
        data = self.buffer.read()
        self.fs.pipe_file(self.path, data)