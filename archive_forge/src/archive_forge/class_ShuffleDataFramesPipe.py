import random
from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes.datapipe import DFIterDataPipe, IterDataPipe
from torch.utils.data.datapipes.dataframe import dataframe_wrapper as df_wrapper
@functional_datapipe('_dataframes_shuffle', enable_df_api_tracing=True)
class ShuffleDataFramesPipe(DFIterDataPipe):

    def __init__(self, source_datapipe):
        self.source_datapipe = source_datapipe

    def __iter__(self):
        size = None
        all_buffer = []
        for df in self.source_datapipe:
            if size is None:
                size = df_wrapper.get_len(df)
            for i in range(df_wrapper.get_len(df)):
                all_buffer.append(df_wrapper.get_item(df, i))
        random.shuffle(all_buffer)
        buffer = []
        for df in all_buffer:
            buffer.append(df)
            if len(buffer) == size:
                yield df_wrapper.concat(buffer)
                buffer = []
        if len(buffer):
            yield df_wrapper.concat(buffer)