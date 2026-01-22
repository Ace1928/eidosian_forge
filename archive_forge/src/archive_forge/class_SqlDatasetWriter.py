import multiprocessing
from typing import TYPE_CHECKING, Optional, Union
from .. import Dataset, Features, config
from ..formatting import query_table
from ..packaged_modules.sql.sql import Sql
from ..utils import tqdm as hf_tqdm
from .abc import AbstractDatasetInputStream
class SqlDatasetWriter:

    def __init__(self, dataset: Dataset, name: str, con: Union[str, 'sqlalchemy.engine.Connection', 'sqlalchemy.engine.Engine', 'sqlite3.Connection'], batch_size: Optional[int]=None, num_proc: Optional[int]=None, **to_sql_kwargs):
        if num_proc is not None and num_proc <= 0:
            raise ValueError(f'num_proc {num_proc} must be an integer > 0.')
        self.dataset = dataset
        self.name = name
        self.con = con
        self.batch_size = batch_size if batch_size else config.DEFAULT_MAX_BATCH_SIZE
        self.num_proc = num_proc
        self.to_sql_kwargs = to_sql_kwargs

    def write(self) -> int:
        _ = self.to_sql_kwargs.pop('sql', None)
        _ = self.to_sql_kwargs.pop('con', None)
        index = self.to_sql_kwargs.pop('index', False)
        written = self._write(index=index, **self.to_sql_kwargs)
        return written

    def _batch_sql(self, args):
        offset, index, to_sql_kwargs = args
        to_sql_kwargs = {**to_sql_kwargs, 'if_exists': 'append'} if offset > 0 else to_sql_kwargs
        batch = query_table(table=self.dataset.data, key=slice(offset, offset + self.batch_size), indices=self.dataset._indices)
        df = batch.to_pandas()
        num_rows = df.to_sql(self.name, self.con, index=index, **to_sql_kwargs)
        return num_rows or len(df)

    def _write(self, index, **to_sql_kwargs) -> int:
        """Writes the pyarrow table as SQL to a database.

        Caller is responsible for opening and closing the SQL connection.
        """
        written = 0
        if self.num_proc is None or self.num_proc == 1:
            for offset in hf_tqdm(range(0, len(self.dataset), self.batch_size), unit='ba', desc='Creating SQL from Arrow format'):
                written += self._batch_sql((offset, index, to_sql_kwargs))
        else:
            num_rows, batch_size = (len(self.dataset), self.batch_size)
            with multiprocessing.Pool(self.num_proc) as pool:
                for num_rows in hf_tqdm(pool.imap(self._batch_sql, [(offset, index, to_sql_kwargs) for offset in range(0, num_rows, batch_size)]), total=num_rows // batch_size + 1 if num_rows % batch_size else num_rows // batch_size, unit='ba', desc='Creating SQL from Arrow format'):
                    written += num_rows
        return written