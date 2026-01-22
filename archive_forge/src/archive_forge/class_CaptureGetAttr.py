from typing import Any, Dict, List
from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes.datapipe import DFIterDataPipe, IterDataPipe
from torch.utils.data.datapipes.dataframe.structures import DataChunkDF
class CaptureGetAttr(Capture):

    def __init__(self, src, name, ctx):
        self.ctx = ctx
        self.src = src
        self.name = name

    def __str__(self):
        return f'{self.src}.{self.name}'

    def execute(self):
        val = get_val(self.src)
        return getattr(val, self.name)