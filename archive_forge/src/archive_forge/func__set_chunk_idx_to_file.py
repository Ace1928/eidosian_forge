from parlai.core.teachers import DialogTeacher, ChunkTeacher, ChunkOutput
from parlai.core.message import Message
from .build import build
import json
import os
from typing import List, Tuple
def _set_chunk_idx_to_file(self):
    folder = self._get_data_folder()
    all_subdirs = sorted([x for x in os.listdir(folder) if 'README' not in x])
    self.chunk_idx_to_file = {i: x for i, x in enumerate(all_subdirs)}