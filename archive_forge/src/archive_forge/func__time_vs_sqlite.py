from qpd_test.tests_base import TestsBase
from typing import Any, Tuple, Iterable
from pandasql import sqldf
from timeit import timeit
def _time_vs_sqlite(self, n: int, sql: str, **dfs: Any) -> Tuple[float, float]:
    data: Any = self.to_dfs(dfs)
    qpd_time = timeit(lambda: self._run(sql, data), number=n) / n
    data = self.to_pd_dfs(dfs)
    sqlite_time = timeit(lambda: sqldf(sql, data), number=n) / n
    return (qpd_time, sqlite_time)