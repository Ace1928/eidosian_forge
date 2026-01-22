from typing import Any, Dict, List
from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes.datapipe import DFIterDataPipe, IterDataPipe
from torch.utils.data.datapipes.dataframe.structures import DataChunkDF
class CaptureSetItem(Capture):

    def __init__(self, left, key, value, ctx):
        self.ctx = ctx
        self.left = left
        self.key = key
        self.value = value

    def __str__(self):
        return f'{self.left}[{get_val(self.key)}] = {self.value}'

    def execute(self):
        left = self.left.execute()
        value = self.value.execute()
        left[self.key] = value