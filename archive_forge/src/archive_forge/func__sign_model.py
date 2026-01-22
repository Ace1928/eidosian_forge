import collections
import gzip
import io
import logging
import struct
import numpy as np
def _sign_model(fout):
    """
    Write signature of the file in Facebook's native fastText `.bin` format
    to the binary output stream `fout`. Signature includes magic bytes and version.

    Name mimics original C++ implementation, see
    [FastText::signModel](https://github.com/facebookresearch/fastText/blob/master/src/fasttext.cc)

    Parameters
    ----------
    fout: writeable binary stream
    """
    fout.write(_FASTTEXT_FILEFORMAT_MAGIC.tobytes())
    fout.write(_FASTTEXT_VERSION.tobytes())