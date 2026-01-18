from qpd_test.tests_base import TestsBase
import pandas as pd
def test_window_min_max(self):
    for func in ['MIN', 'MAX']:
        a = self.make_rand_df(100, a=float, b=(int, 50), c=(str, 50))
        self.eq_sqlite(f'\n                    SELECT a,b,\n                        {func}(b) OVER () AS a1,\n                        {func}(b) OVER (PARTITION BY c) AS a2,\n                        {func}(b+a) OVER (PARTITION BY c,b) AS a3,\n                        {func}(b+a) OVER (PARTITION BY b ORDER BY a\n                            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS a4,\n                        {func}(b+a) OVER (PARTITION BY b ORDER BY a DESC\n                            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS a5,\n                        {func}(b+a) OVER (PARTITION BY b ORDER BY a\n                            ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING)\n                            AS a6\n                    FROM a\n                    ', a=a)
        if pd.__version__ >= '1.1':
            self.eq_sqlite(f'\n                        SELECT a,b,\n                            {func}(b) OVER (ORDER BY a DESC\n                                ROWS BETWEEN 2 PRECEDING AND 1 PRECEDING) AS a6,\n                            {func}(b) OVER (ORDER BY a DESC\n                                ROWS BETWEEN 2 PRECEDING AND 1 FOLLOWING) AS a7,\n                            {func}(b) OVER (ORDER BY a DESC\n                                ROWS BETWEEN 2 PRECEDING AND UNBOUNDED FOLLOWING) AS a8\n                        FROM a\n                        ', a=a)
        if pd.__version__ < '1.1':
            b = self.make_rand_df(10, a=float, b=(int, 0), c=(str, 0))
            self.eq_sqlite(f'\n                        SELECT a,b,\n                            {func}(b) OVER (PARTITION BY b ORDER BY a DESC\n                                ROWS BETWEEN 2 PRECEDING AND 1 PRECEDING) AS a6\n                        FROM a\n                        ', a=b)