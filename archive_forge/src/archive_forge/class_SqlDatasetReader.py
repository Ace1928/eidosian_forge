import multiprocessing
from typing import TYPE_CHECKING, Optional, Union
from .. import Dataset, Features, config
from ..formatting import query_table
from ..packaged_modules.sql.sql import Sql
from ..utils import tqdm as hf_tqdm
from .abc import AbstractDatasetInputStream
class SqlDatasetReader(AbstractDatasetInputStream):

    def __init__(self, sql: Union[str, 'sqlalchemy.sql.Selectable'], con: Union[str, 'sqlalchemy.engine.Connection', 'sqlalchemy.engine.Engine', 'sqlite3.Connection'], features: Optional[Features]=None, cache_dir: str=None, keep_in_memory: bool=False, **kwargs):
        super().__init__(features=features, cache_dir=cache_dir, keep_in_memory=keep_in_memory, **kwargs)
        self.builder = Sql(cache_dir=cache_dir, features=features, sql=sql, con=con, **kwargs)

    def read(self):
        download_config = None
        download_mode = None
        verification_mode = None
        base_path = None
        self.builder.download_and_prepare(download_config=download_config, download_mode=download_mode, verification_mode=verification_mode, base_path=base_path)
        dataset = self.builder.as_dataset(split='train', verification_mode=verification_mode, in_memory=self.keep_in_memory)
        return dataset