import random
from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes.datapipe import DFIterDataPipe, IterDataPipe
from torch.utils.data.datapipes.dataframe import dataframe_wrapper as df_wrapper
@functional_datapipe('_dataframes_filter', enable_df_api_tracing=True)
class FilterDataFramesPipe(DFIterDataPipe):

    def __init__(self, source_datapipe, filter_fn):
        self.source_datapipe = source_datapipe
        self.filter_fn = filter_fn

    def __iter__(self):
        size = None
        all_buffer = []
        filter_res = []
        for df in self.source_datapipe:
            if size is None:
                size = len(df.index)
            for i in range(len(df.index)):
                all_buffer.append(df[i:i + 1])
                filter_res.append(self.filter_fn(df.iloc[i]))
        buffer = []
        for df, res in zip(all_buffer, filter_res):
            if res:
                buffer.append(df)
                if len(buffer) == size:
                    yield df_wrapper.concat(buffer)
                    buffer = []
        if len(buffer):
            yield df_wrapper.concat(buffer)