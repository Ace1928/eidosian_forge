from typing import Optional
import gzip
import os
from parlai.core.build_data import DownloadableFile
import parlai.core.build_data as build_data
@classmethod
def wids2sent(cls, wids, d_vocab):
    return ' '.join([d_vocab[w] for w in wids])