from qpd_test.tests_base import TestsBase
import pandas as pd
def test_window_sum_avg(self):
    a = self.make_rand_df(100, a=float, b=int, c=(str, 50))
    for func in ['SUM', 'AVG']:
        self.eq_sqlite(f'\n                    SELECT a,b,\n                        {func}(b) OVER () AS a1,\n                        {func}(b) OVER (PARTITION BY c) AS a2,\n                        {func}(b+a) OVER (PARTITION BY c,b) AS a3,\n                        {func}(b+a) OVER (PARTITION BY b ORDER BY a\n                            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS a4,\n                        {func}(b+a) OVER (PARTITION BY b ORDER BY a DESC\n                            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS a5,\n                        {func}(b+a) OVER (PARTITION BY b ORDER BY a\n                            ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING)\n                            AS a6\n                    FROM a\n                    ', a=a)