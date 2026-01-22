from qpd_test.tests_base import TestsBase
from typing import Any, Tuple, Iterable
from pandasql import sqldf
from timeit import timeit
def _test_select_where(self, *sizes: int, repeat: int=1) -> Iterable[Tuple[int, float, float]]:
    for n in sizes:
        a = self.make_rand_df(n, a=int, b=str, c=float, d=int, e=bool, f=str, g=str, h=float, i=str, j=int, k=bool, l=float)
        q, s = self._time_vs_sqlite(repeat, '\n                    SELECT l AS ll,k AS kk,j AS jj,h AS hh,f,e,b,(a+l)*(a-l) AS a FROM a\n                    WHERE l<10 OR e OR (c<0.5 AND h>0.5)\n                    ', a=a)
        yield (n, q, s)