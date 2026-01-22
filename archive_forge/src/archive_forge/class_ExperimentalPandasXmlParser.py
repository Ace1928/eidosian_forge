import warnings
from io import BytesIO
import pandas
from pandas.util._decorators import doc
from modin.core.io.file_dispatcher import OpenFile
from modin.core.storage_formats.pandas.parsers import (
@doc(_doc_pandas_parser_class, data_type='XML files')
class ExperimentalPandasXmlParser(PandasParser):

    @staticmethod
    @doc(_doc_parse_func, parameters=_doc_parse_parameters_common)
    def parse(fname, **kwargs):
        warnings.filterwarnings('ignore')
        num_splits = 1
        single_worker_read = kwargs.pop('single_worker_read', None)
        df = pandas.read_xml(fname, **kwargs)
        if single_worker_read:
            return df
        length = len(df)
        width = len(df.columns)
        return _split_result_for_readers(1, num_splits, df) + [length, width]