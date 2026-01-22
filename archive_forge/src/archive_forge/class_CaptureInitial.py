from typing import Any, Dict, List
from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes.datapipe import DFIterDataPipe, IterDataPipe
from torch.utils.data.datapipes.dataframe.structures import DataChunkDF
class CaptureInitial(CaptureVariable):

    def __init__(self, schema_df=None):
        new_ctx: Dict[str, List[Any]] = {'operations': [], 'variables': [], 'schema_df': schema_df}
        super().__init__(None, new_ctx)
        self.name = f'input_{self.name}'