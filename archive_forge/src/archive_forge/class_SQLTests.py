from qpd_test.tests_base import TestsBase
import pandas as pd
class SQLTests(object):

    class Tests(TestsBase):

        def test_basic_select_from(self):
            df = self.make_rand_df(5, a=(int, 2), b=(str, 3), c=(float, 4))
            self.eq_sqlite("SELECT 1 AS a, 1.5 AS b, 'x' AS c")
            self.eq_sqlite("SELECT 1+2 AS a, 1.5*3 AS b, 'x' AS c")
            self.eq_sqlite('SELECT * FROM a', a=df)
            self.eq_sqlite('SELECT * FROM a AS x', a=df)
            self.eq_sqlite('SELECT b AS bb, a+1-2*3.0/4 AS cc, x.* FROM a AS x', a=df)
            self.eq_sqlite("SELECT *, 1 AS x, 2.5 AS y, 'z' AS z FROM a AS x", a=df)
            self.eq_sqlite('SELECT *, -(1.0+a)/3 AS x, +(2.5) AS y FROM a AS x', a=df)

        def test_basic_select_from_special_chars(self):
            df = self.make_rand_df(5, **{'a b': (int, 2), '-': (str, 3), 'c': (float, 4)})
            self.eq_sqlite("SELECT 1 AS `a c`, 1.5 AS `-`, 'x' AS c")
            self.eq_sqlite("SELECT 1+2 AS `a b`, 1.5*3 AS b, 'x' AS c")
            self.eq_sqlite('SELECT *, 1 AS `c d` FROM a', a=df)
            self.eq_sqlite('SELECT * FROM a AS x', a=df)
            self.eq_sqlite('SELECT `-` AS `b b `, `a b`+1-2*3.0/4 AS `cc`, x.* FROM a AS x', a=df)
            self.eq_sqlite("SELECT *, 1 AS x, 2.5 AS y, 'z' AS z FROM a AS x", a=df)
            self.eq_sqlite('SELECT *, -(1.0+`a b`)/3 AS x, +(2.5) AS y FROM a AS x', a=df)

        def test_case_when(self):
            a = self.make_rand_df(100, a=(int, 20), b=(str, 30), c=(float, 40))
            self.eq_sqlite('\n                SELECT a,b,c,\n                    CASE\n                        WHEN a<10 THEN a+3\n                        WHEN c<0.5 THEN a+5\n                        ELSE (1+2)*3 + a\n                    END AS d\n                FROM a\n                ', a=a)

        def test_drop_duplicates(self):
            a = self.make_rand_df(100, a=int, b=int)
            self.eq_sqlite('\n                SELECT DISTINCT b, a FROM a\n                ', a=a)
            a = self.make_rand_df(100, a=(int, 50), b=(int, 50))
            self.eq_sqlite('\n                SELECT DISTINCT b, a FROM a\n                ', a=a)
            a = self.make_rand_df(100, a=(int, 50), b=(str, 50), c=float)
            self.eq_sqlite('\n                SELECT DISTINCT b, a FROM a\n                ', a=a)

        def test_order_by_no_limit(self):
            a = self.make_rand_df(100, a=(int, 50), b=(str, 50), c=float)
            self.eq_sqlite('\n                SELECT DISTINCT b, a FROM a ORDER BY a\n                ', a=a)

        def test_order_by_limit(self):
            a = self.make_rand_df(100, a=(int, 50), b=(str, 50), c=float)
            self.eq_sqlite('\n                SELECT DISTINCT b, a FROM a LIMIT 0\n                ', a=a)
            self.eq_sqlite('\n                SELECT DISTINCT b, a FROM a ORDER BY a LIMIT 2\n                ', a=a)
            self.eq_sqlite('\n                SELECT b, a FROM a\n                    ORDER BY a NULLS LAST, b NULLS FIRST LIMIT 10\n                ', a=a)

        def test_where(self):
            df = self.make_rand_df(100, a=(int, 30), b=(str, 30), c=(float, 30))
            self.eq_sqlite('SELECT * FROM a WHERE TRUE OR TRUE', a=df)
            self.eq_sqlite('SELECT * FROM a WHERE TRUE AND TRUE', a=df)
            self.eq_sqlite('SELECT * FROM a WHERE FALSE OR FALSE', a=df)
            self.eq_sqlite('SELECT * FROM a WHERE FALSE AND FALSE', a=df)
            self.eq_sqlite("SELECT * FROM a WHERE TRUE OR b<='ssssss8'", a=df)
            self.eq_sqlite("SELECT * FROM a WHERE TRUE AND b<='ssssss8'", a=df)
            self.eq_sqlite("SELECT * FROM a WHERE FALSE OR b<='ssssss8'", a=df)
            self.eq_sqlite("SELECT * FROM a WHERE FALSE AND b<='ssssss8'", a=df)
            self.eq_sqlite("SELECT * FROM a WHERE a==10 OR b<='ssssss8'", a=df)
            self.eq_sqlite('SELECT * FROM a WHERE c IS NOT NULL OR (a<5 AND b IS NOT NULL)', a=df)
            df = self.make_rand_df(100, a=(float, 30), b=(float, 30), c=(float, 30))
            self.eq_sqlite('SELECT * FROM a WHERE a<0.5 AND b<0.5 AND c<0.5', a=df)
            self.eq_sqlite('SELECT * FROM a WHERE a<0.5 OR b<0.5 AND c<0.5', a=df)
            self.eq_sqlite('SELECT * FROM a WHERE a IS NULL OR (b<0.5 AND c<0.5)', a=df)
            self.eq_sqlite('SELECT * FROM a WHERE a*b IS NULL OR (b*c<0.5 AND c*a<0.5)', a=df)

        def test_in_between(self):
            df = self.make_rand_df(10, a=(int, 3), b=(str, 3))
            self.eq_sqlite('SELECT * FROM a WHERE a IN (2,4,6)', a=df)
            self.eq_sqlite('SELECT * FROM a WHERE a BETWEEN 2 AND 4+1', a=df)
            self.eq_sqlite('SELECT * FROM a WHERE a NOT IN (2,4,6)', a=df)
            self.eq_sqlite('SELECT * FROM a WHERE a NOT BETWEEN 2 AND 4+1', a=df)

        def test_join_inner(self):
            a = self.make_rand_df(100, a=(int, 40), b=(str, 40), c=(float, 40))
            b = self.make_rand_df(80, d=(float, 10), a=(int, 10), b=(str, 10))
            self.eq_sqlite('SELECT a.*, d, d*c AS x FROM a INNER JOIN b ON a.a=b.a AND a.b=b.b', a=a, b=b)

        def test_join_inner_special_chars(self):
            a = self.make_rand_df(100, **{'a b': (int, 40), 'b': (str, 40), 'c': (float, 40)})
            b = self.make_rand_df(80, **{'d': (float, 10), 'a b': (int, 10), 'b': (str, 10)})
            self.eq_sqlite('SELECT a.*, d, d*c AS x FROM a INNER JOIN b ON a.`a b`=b.`a b` AND a.b=b.b', a=a, b=b)

        def test_join_left(self):
            a = self.make_rand_df(100, a=(int, 40), b=(str, 40), c=(float, 40))
            b = self.make_rand_df(80, d=(float, 10), a=(int, 10), b=(str, 10))
            self.eq_sqlite('SELECT a.*, d, d*c AS x FROM a LEFT JOIN b ON a.a=b.a AND a.b=b.b', a=a, b=b)

        def test_join_right(self):
            a = ([[0, 1], [None, 3]], ['a', 'b'])
            b = ([[0, 10], [None, 30]], ['a', 'c'])
            self.assert_eq(dict(a=a, b=b), 'SELECT a.*,c FROM a RIGHT JOIN b ON a.a=b.a', [[0, 1, 10], [None, None, 30]], ['a', 'b', 'c'])

        def test_join_full(self):
            a = ([[0, 1], [None, 3]], ['a', 'b'])
            b = ([[0, 10], [None, 30]], ['a', 'c'])
            self.assert_eq(dict(a=a, b=b), 'SELECT a.*,c FROM a FULL JOIN b ON a.a=b.a', [[0, 1, 10], [None, 3, None], [None, None, 30]], ['a', 'b', 'c'])

        def test_join_cross(self):
            a = self.make_rand_df(10, a=(int, 4), b=(str, 4), c=(float, 4))
            b = self.make_rand_df(20, dd=(float, 1), aa=(int, 1), bb=(str, 1))
            self.eq_sqlite('SELECT * FROM a CROSS JOIN b', a=a, b=b)

        def test_join_semi(self):
            a = ([[0, 1], [None, 3]], ['a', 'b'])
            b = ([[0, 10], [None, 30]], ['a', 'b'])
            self.assert_eq(dict(a=a, b=b), 'SELECT * FROM a LEFT SEMI JOIN b ON a.a=b.a', [[0, 1]], ['a', 'b'])
            self.assert_eq(dict(a=a, b=b), 'SELECT a.* FROM a LEFT SEMI JOIN b ON a.a=b.a', [[0, 1]], ['a', 'b'])

        def test_join_anti(self):
            a = ([[0, 1], [None, 3]], ['a', 'b'])
            b = ([[0, 10], [None, 30]], ['a', 'b'])
            self.assert_eq(dict(a=a, b=b), 'SELECT * FROM a LEFT ANTI JOIN b ON a.a=b.a', [[None, 3]], ['a', 'b'])
            self.assert_eq(dict(a=a, b=b), 'SELECT a.* FROM a LEFT ANTI JOIN b ON a.a=b.a', [[None, 3]], ['a', 'b'])

        def test_join_multi(self):
            a = self.make_rand_df(100, a=(int, 40), b=(str, 40), c=(float, 40))
            b = self.make_rand_df(80, d=(float, 10), a=(int, 10), b=(str, 10))
            c = self.make_rand_df(80, dd=(float, 10), a=(int, 10), b=(str, 10))
            self.eq_sqlite('\n                SELECT a.*,d,dd FROM a\n                    INNER JOIN b ON a.a=b.a AND a.b=b.b\n                    INNER JOIN c ON a.a=c.a AND c.b=b.b\n                ', a=a, b=b, c=c)

        def test_agg_count_no_group_by(self):
            a = self.make_rand_df(100, a=(int, 50), b=(str, 50), c=(int, 30), d=(str, 40), e=(float, 40))
            self.eq_sqlite('\n                SELECT\n                    COUNT(a) AS c_a,\n                    COUNT(DISTINCT a) AS cd_a,\n                    COUNT(b) AS c_b,\n                    COUNT(DISTINCT b) AS cd_b,\n                    COUNT(c) AS c_c,\n                    COUNT(DISTINCT c) AS cd_c,\n                    COUNT(d) AS c_d,\n                    COUNT(DISTINCT d) AS cd_d,\n                    COUNT(e) AS c_e,\n                    COUNT(DISTINCT a) AS cd_e\n                FROM a\n                ', a=a)
            b = ([[1, 'x', 1.5], [2, None, 2.5], [2, None, 2.5]], ['a', 'b', 'c'])
            self.assert_eq(dict(a=a, b=b), '\n                SELECT\n                    COUNT(*) AS a1,\n                    COUNT(DISTINCT *) AS a2,\n                    COUNT(a, b) AS a3,\n                    COUNT(DISTINCT a,b) AS a4,\n                    COUNT(a, b) + COUNT(DISTINCT a,b) AS a5\n                FROM b\n                ', [[3, 2, 3, 2, 5]], ['a1', 'a2', 'a3', 'a4', 'a5'])

        def test_agg_count(self):
            a = self.make_rand_df(100, a=(int, 50), b=(str, 50), c=(int, 30), d=(str, 40), e=(float, 40))
            self.eq_sqlite('\n                SELECT\n                    a, b, a+1 AS c,\n                    COUNT(c) AS c_c,\n                    COUNT(DISTINCT c) AS cd_c,\n                    COUNT(d) AS c_d,\n                    COUNT(DISTINCT d) AS cd_d,\n                    COUNT(e) AS c_e,\n                    COUNT(DISTINCT a) AS cd_e\n                FROM a GROUP BY a, b\n                ', a=a)
            b = ([[1, 'x', 1.5], [2, None, 2.5], [2, None, 2.5]], ['a', 'b', 'c'])
            self.assert_eq(dict(a=a, b=b), '\n                SELECT\n                    a, b,\n                    COUNT(*) AS a1,\n                    COUNT(DISTINCT *) AS a2,\n                    COUNT(c) AS a3,\n                    COUNT(DISTINCT c) AS a4,\n                    COUNT(c) + COUNT(DISTINCT c) AS a5\n                FROM b GROUP BY a, b\n                ', [[1, 'x', 1, 1, 1, 1, 2], [2, None, 2, 1, 2, 1, 3]], ['a', 'b', 'a1', 'a2', 'a3', 'a4', 'a5'])

        def test_agg_sum_avg_no_group_by(self):
            self.eq_sqlite('\n                SELECT\n                    SUM(a) AS sum_a,\n                    AVG(a) AS avg_a\n                FROM a\n                ', a=([[float('nan')]], ['a']))
            a = self.make_rand_df(100, a=(int, 50), b=(str, 50), c=(int, 30), d=(str, 40), e=(float, 40))
            self.eq_sqlite('\n                SELECT\n                    SUM(a) AS sum_a,\n                    AVG(a) AS avg_a,\n                    SUM(c) AS sum_c,\n                    AVG(c) AS avg_c,\n                    SUM(e) AS sum_e,\n                    AVG(e) AS avg_e,\n                    SUM(a)+AVG(e) AS mix_1,\n                    SUM(a+e) AS mix_2\n                FROM a\n                ', a=a)

        def test_agg_sum_avg(self):
            a = self.make_rand_df(100, a=(int, 50), b=(str, 50), c=(int, 30), d=(str, 40), e=(float, 40))
            self.eq_sqlite('\n                SELECT\n                    a,b, a+1 AS c,\n                    SUM(c) AS sum_c,\n                    AVG(c) AS avg_c,\n                    SUM(e) AS sum_e,\n                    AVG(e) AS avg_e,\n                    SUM(a)+AVG(e) AS mix_1,\n                    SUM(a+e) AS mix_2\n                FROM a GROUP BY a,b\n                ', a=a)

        def test_agg_min_max_no_group_by(self):
            a = self.make_rand_df(100, a=(int, 50), b=(str, 50), c=(int, 30), d=(str, 40), e=(float, 40))
            self.eq_sqlite('\n                SELECT\n                    MIN(a) AS min_a,\n                    MAX(a) AS max_a,\n                    MIN(b) AS min_b,\n                    MAX(b) AS max_b,\n                    MIN(c) AS min_c,\n                    MAX(c) AS max_c,\n                    MIN(d) AS min_d,\n                    MAX(d) AS max_d,\n                    MIN(e) AS min_e,\n                    MAX(e) AS max_e,\n                    MIN(a+e) AS mix_1,\n                    MIN(a)+MIN(e) AS mix_2\n                FROM a\n                ', a=a)

        def test_agg_min_max(self):
            a = self.make_rand_df(100, a=(int, 50), b=(str, 50), c=(int, 30), d=(str, 40), e=(float, 40))
            self.eq_sqlite('\n                SELECT\n                    a, b, a+1 AS c,\n                    MIN(c) AS min_c,\n                    MAX(c) AS max_c,\n                    MIN(d) AS min_d,\n                    MAX(d) AS max_d,\n                    MIN(e) AS min_e,\n                    MAX(e) AS max_e,\n                    MIN(a+e) AS mix_1,\n                    MIN(a)+MIN(e) AS mix_2\n                FROM a GROUP BY a, b\n                ', a=a)

        def test_agg_first_last_no_group_by(self):
            a = ([[1, 'x', None], [2, None, 2.5], [2, None, 2.5]], ['a', 'b', 'c'])
            self.assert_eq(dict(a=a), '\n                SELECT\n                    FIRST(a) AS a1,\n                    LAST(a) AS a2,\n                    FIRST_VALUE(b) AS a3,\n                    LAST_VALUE(b) AS a4,\n                    FIRST_VALUE(c) AS a5,\n                    LAST_VALUE(c) AS a6\n                FROM a\n                ', [[1, 2, 'x', None, None, 2.5]], ['a1', 'a2', 'a3', 'a4', 'a5', 'a6'])

        def test_agg_first_last(self):
            a = ([[1, 'x', None], [2, None, 3.5], [2, 1, 2.5]], ['a', 'b', 'c'])
            self.assert_eq(dict(a=a), '\n                SELECT\n                    a,\n                    FIRST(b) AS a1,\n                    LAST(b) AS a2,\n                    FIRST_VALUE(c) AS a3,\n                    LAST_VALUE(c) AS a4\n                FROM a GROUP BY a\n                ', [[1, 'x', 'x', None, None], [2, None, 1, 3.5, 2.5]], ['a', 'a1', 'a2', 'a3', 'a4'])

        def test_window_row_number(self):
            a = self.make_rand_df(100, a=int, b=(float, 50))
            self.eq_sqlite('\n                SELECT *,\n                    ROW_NUMBER() OVER (ORDER BY a ASC, b DESC NULLS FIRST) AS a1,\n                    ROW_NUMBER() OVER (ORDER BY a ASC, b DESC NULLS LAST) AS a2,\n                    ROW_NUMBER() OVER (ORDER BY a ASC, b ASC NULLS FIRST) AS a3,\n                    ROW_NUMBER() OVER (ORDER BY a ASC, b ASC NULLS LAST) AS a4,\n                    ROW_NUMBER() OVER (PARTITION BY a ORDER BY a,b DESC) AS a5\n                FROM a\n                ', a=a)
            a = self.make_rand_df(100, a=(int, 50), b=(str, 50), c=(int, 30), d=(str, 40), e=float)
            self.eq_sqlite('\n                SELECT *,\n                    ROW_NUMBER() OVER (ORDER BY a ASC, b DESC NULLS FIRST, e) AS a1,\n                    ROW_NUMBER() OVER (ORDER BY a ASC, b DESC NULLS LAST, e) AS a2,\n                    ROW_NUMBER() OVER (PARTITION BY a ORDER BY a,b DESC, e) AS a3,\n                    ROW_NUMBER() OVER (PARTITION BY a,c ORDER BY a,b DESC, e) AS a4\n                FROM a\n                ', a=a)

        def test_window_row_number_partition_by(self):
            a = self.make_rand_df(100, a=int, b=(float, 50))
            self.eq_sqlite('\n                SELECT *,\n                    ROW_NUMBER() OVER (PARTITION BY a ORDER BY a,b DESC) AS a5\n                FROM a\n                ', a=a)
            a = self.make_rand_df(100, a=(int, 50), b=(str, 50), c=(int, 30), d=(str, 40), e=float)
            self.eq_sqlite('\n                SELECT *,\n                    ROW_NUMBER() OVER (PARTITION BY a ORDER BY a,b DESC, e) AS a3,\n                    ROW_NUMBER() OVER (PARTITION BY a,c ORDER BY a,b DESC, e) AS a4\n                FROM a\n                ', a=a)

        def test_window_ranks(self):
            a = self.make_rand_df(100, a=int, b=(float, 50), c=(str, 50))
            self.eq_sqlite('\n                SELECT *,\n                    RANK() OVER (PARTITION BY a ORDER BY b DESC NULLS FIRST, c) AS a1,\n                    DENSE_RANK() OVER (ORDER BY a ASC, b DESC NULLS LAST, c DESC) AS a2,\n                    PERCENT_RANK() OVER (ORDER BY a ASC, b ASC NULLS LAST, c) AS a4\n                FROM a\n                ', a=a)

        def test_window_ranks_partition_by(self):
            a = self.make_rand_df(100, a=int, b=(float, 50), c=(str, 50))
            self.eq_sqlite('\n                SELECT *,\n                    RANK() OVER (PARTITION BY a ORDER BY b DESC NULLS FIRST, c) AS a1,\n                    DENSE_RANK() OVER\n                        (PARTITION BY a ORDER BY a ASC, b DESC NULLS LAST, c DESC)\n                        AS a2,\n                    PERCENT_RANK() OVER\n                        (PARTITION BY a ORDER BY a ASC, b ASC NULLS LAST, c) AS a4\n                FROM a\n                ', a=a)

        def test_window_lead_lag(self):
            a = self.make_rand_df(100, a=float, b=(int, 50), c=(str, 50))
            self.eq_sqlite('\n                SELECT\n                    LEAD(b,1) OVER (ORDER BY a) AS a1,\n                    LEAD(b,2,10) OVER (ORDER BY a) AS a2,\n                    LEAD(b,1) OVER (PARTITION BY c ORDER BY a) AS a3,\n                    LEAD(b,1) OVER (PARTITION BY c ORDER BY b, a ASC NULLS LAST) AS a5,\n\n                    LAG(b,1) OVER (ORDER BY a) AS b1,\n                    LAG(b,2,10) OVER (ORDER BY a) AS b2,\n                    LAG(b,1) OVER (PARTITION BY c ORDER BY a) AS b3,\n                    LAG(b,1) OVER (PARTITION BY c ORDER BY b, a ASC NULLS LAST) AS b5\n                FROM a\n                ', a=a)

        def test_window_lead_lag_partition_by(self):
            a = self.make_rand_df(100, a=float, b=(int, 50), c=(str, 50))
            self.eq_sqlite('\n                SELECT\n                    LEAD(b,1,10) OVER (PARTITION BY c ORDER BY a) AS a3,\n                    LEAD(b,1) OVER (PARTITION BY c ORDER BY b, a ASC NULLS LAST) AS a5,\n\n                    LAG(b,1) OVER (PARTITION BY c ORDER BY a) AS b3,\n                    LAG(b,1) OVER (PARTITION BY c ORDER BY b, a ASC NULLS LAST) AS b5\n                FROM a\n                ', a=a)

        def test_window_sum_avg(self):
            a = self.make_rand_df(100, a=float, b=int, c=(str, 50))
            for func in ['SUM', 'AVG']:
                self.eq_sqlite(f'\n                    SELECT a,b,\n                        {func}(b) OVER () AS a1,\n                        {func}(b) OVER (PARTITION BY c) AS a2,\n                        {func}(b+a) OVER (PARTITION BY c,b) AS a3,\n                        {func}(b+a) OVER (PARTITION BY b ORDER BY a\n                            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS a4,\n                        {func}(b+a) OVER (PARTITION BY b ORDER BY a DESC\n                            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS a5,\n                        {func}(b+a) OVER (PARTITION BY b ORDER BY a\n                            ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING)\n                            AS a6\n                    FROM a\n                    ', a=a)

        def test_window_sum_avg_partition_by(self):
            a = self.make_rand_df(100, a=float, b=int, c=(str, 50))
            for func in ['SUM', 'AVG']:
                self.eq_sqlite(f'\n                    SELECT a,b,\n                        {func}(b+a) OVER (PARTITION BY c,b) AS a3,\n                        {func}(b+a) OVER (PARTITION BY b ORDER BY a\n                            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS a4,\n                        {func}(b+a) OVER (PARTITION BY b ORDER BY a DESC\n                            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS a5,\n                        {func}(b+a) OVER (PARTITION BY b ORDER BY a\n                            ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING)\n                            AS a6\n                    FROM a\n                    ', a=a)

        def test_window_min_max(self):
            for func in ['MIN', 'MAX']:
                a = self.make_rand_df(100, a=float, b=(int, 50), c=(str, 50))
                self.eq_sqlite(f'\n                    SELECT a,b,\n                        {func}(b) OVER () AS a1,\n                        {func}(b) OVER (PARTITION BY c) AS a2,\n                        {func}(b+a) OVER (PARTITION BY c,b) AS a3,\n                        {func}(b+a) OVER (PARTITION BY b ORDER BY a\n                            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS a4,\n                        {func}(b+a) OVER (PARTITION BY b ORDER BY a DESC\n                            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS a5,\n                        {func}(b+a) OVER (PARTITION BY b ORDER BY a\n                            ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING)\n                            AS a6\n                    FROM a\n                    ', a=a)
                if pd.__version__ >= '1.1':
                    self.eq_sqlite(f'\n                        SELECT a,b,\n                            {func}(b) OVER (ORDER BY a DESC\n                                ROWS BETWEEN 2 PRECEDING AND 1 PRECEDING) AS a6,\n                            {func}(b) OVER (ORDER BY a DESC\n                                ROWS BETWEEN 2 PRECEDING AND 1 FOLLOWING) AS a7,\n                            {func}(b) OVER (ORDER BY a DESC\n                                ROWS BETWEEN 2 PRECEDING AND UNBOUNDED FOLLOWING) AS a8\n                        FROM a\n                        ', a=a)
                if pd.__version__ < '1.1':
                    b = self.make_rand_df(10, a=float, b=(int, 0), c=(str, 0))
                    self.eq_sqlite(f'\n                        SELECT a,b,\n                            {func}(b) OVER (PARTITION BY b ORDER BY a DESC\n                                ROWS BETWEEN 2 PRECEDING AND 1 PRECEDING) AS a6\n                        FROM a\n                        ', a=b)

        def test_window_min_max_partition_by(self):
            for func in ['MIN', 'MAX']:
                a = self.make_rand_df(100, a=float, b=(int, 50), c=(str, 50))
                self.eq_sqlite(f'\n                    SELECT a,b,\n                        {func}(b) OVER (PARTITION BY c) AS a2,\n                        {func}(b+a) OVER (PARTITION BY c,b) AS a3,\n                        {func}(b+a) OVER (PARTITION BY b ORDER BY a\n                            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS a4,\n                        {func}(b+a) OVER (PARTITION BY b ORDER BY a DESC\n                            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS a5,\n                        {func}(b+a) OVER (PARTITION BY b ORDER BY a\n                            ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING)\n                            AS a6\n                    FROM a\n                    ', a=a)
                if pd.__version__ < '1.1':
                    b = self.make_rand_df(10, a=float, b=(int, 0), c=(str, 0))
                    self.eq_sqlite(f'\n                        SELECT a,b,\n                            {func}(b) OVER (PARTITION BY b ORDER BY a DESC\n                                ROWS BETWEEN 2 PRECEDING AND 1 PRECEDING) AS a6\n                        FROM a\n                        ', a=b)

        def test_window_count(self):
            for func in ['COUNT']:
                a = self.make_rand_df(100, a=float, b=(int, 50), c=(str, 50))
                self.eq_sqlite(f'\n                    SELECT a,b,\n                        {func}(b) OVER () AS a1,\n                        {func}(b) OVER (PARTITION BY c) AS a2,\n                        {func}(b+a) OVER (PARTITION BY c,b) AS a3,\n                        {func}(b+a) OVER (PARTITION BY b ORDER BY a\n                            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS a4,\n                        {func}(b+a) OVER (PARTITION BY b ORDER BY a DESC\n                            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS a5,\n                        {func}(b+a) OVER (PARTITION BY b ORDER BY a\n                            ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING)\n                            AS a6,\n\n                        {func}(c) OVER () AS b1,\n                        {func}(c) OVER (PARTITION BY c) AS b2,\n                        {func}(c) OVER (PARTITION BY c,b) AS b3,\n                        {func}(c) OVER (PARTITION BY b ORDER BY a\n                            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS b4,\n                        {func}(c) OVER (PARTITION BY b ORDER BY a DESC\n                            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS b5,\n                        {func}(c) OVER (PARTITION BY b ORDER BY a\n                            ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING)\n                            AS b6\n                    FROM a\n                    ', a=a)
                if pd.__version__ >= '1.1':
                    self.eq_sqlite(f'\n                        SELECT a,b,\n                            {func}(b) OVER (ORDER BY a DESC\n                                ROWS BETWEEN 2 PRECEDING AND 0 PRECEDING) AS a6,\n                            {func}(b) OVER (PARTITION BY c ORDER BY a DESC\n                                ROWS BETWEEN 2 PRECEDING AND 0 PRECEDING) AS a9,\n\n                            {func}(c) OVER (ORDER BY a DESC\n                                ROWS BETWEEN 2 PRECEDING AND 0 PRECEDING) AS b6,\n                            {func}(c) OVER (PARTITION BY c ORDER BY a DESC\n                                ROWS BETWEEN 2 PRECEDING AND 0 PRECEDING) AS b9\n                        FROM a\n                        ', a=a)

        def test_window_count_partition_by(self):
            for func in ['COUNT']:
                a = self.make_rand_df(100, a=float, b=(int, 50), c=(str, 50))
                self.eq_sqlite(f'\n                    SELECT a,b,\n                        {func}(b) OVER (PARTITION BY c) AS a2,\n                        {func}(b+a) OVER (PARTITION BY c,b) AS a3,\n                        {func}(b+a) OVER (PARTITION BY b ORDER BY a\n                            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS a4,\n                        {func}(b+a) OVER (PARTITION BY b ORDER BY a DESC\n                            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS a5,\n                        {func}(b+a) OVER (PARTITION BY b ORDER BY a\n                            ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING)\n                            AS a6,\n\n                        {func}(c) OVER (PARTITION BY c) AS b2,\n                        {func}(c) OVER (PARTITION BY c,b) AS b3,\n                        {func}(c) OVER (PARTITION BY b ORDER BY a\n                            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS b4,\n                        {func}(c) OVER (PARTITION BY b ORDER BY a DESC\n                            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS b5,\n                        {func}(c) OVER (PARTITION BY b ORDER BY a\n                            ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING)\n                            AS b6\n                    FROM a\n                    ', a=a)
                if pd.__version__ >= '1.1':
                    self.eq_sqlite(f'\n                        SELECT a,b,\n                            {func}(b) OVER (PARTITION BY c ORDER BY a DESC\n                                ROWS BETWEEN 2 PRECEDING AND 0 PRECEDING) AS a9,\n\n                            {func}(c) OVER (PARTITION BY c ORDER BY a DESC\n                                ROWS BETWEEN 2 PRECEDING AND 0 PRECEDING) AS b9\n                        FROM a\n                        ', a=a)

        def test_nested_query(self):
            a = self.make_rand_df(100, a=float, b=(int, 50), c=(str, 50))
            self.eq_sqlite('\n                SELECT * FROM (\n                SELECT *,\n                    ROW_NUMBER() OVER (PARTITION BY c ORDER BY b, a ASC NULLS LAST) AS r\n                FROM a)\n                WHERE r=1\n                ', a=a)

        def test_union(self):
            a = self.make_rand_df(30, b=(int, 10), c=(str, 10))
            b = self.make_rand_df(80, b=(int, 50), c=(str, 50))
            c = self.make_rand_df(100, b=(int, 50), c=(str, 50))
            self.eq_sqlite('\n                SELECT * FROM a\n                    UNION SELECT * FROM b\n                    UNION SELECT * FROM c\n                ', a=a, b=b, c=c)
            self.eq_sqlite('\n                SELECT * FROM a\n                    UNION ALL SELECT * FROM b\n                    UNION ALL SELECT * FROM c\n                ', a=a, b=b, c=c)

        def test_except(self):
            a = self.make_rand_df(30, b=(int, 10), c=(str, 10))
            b = self.make_rand_df(80, b=(int, 50), c=(str, 50))
            c = self.make_rand_df(100, b=(int, 50), c=(str, 50))
            self.eq_sqlite('\n                SELECT * FROM c\n                    EXCEPT SELECT * FROM b\n                    EXCEPT SELECT * FROM c\n                ', a=a, b=b, c=c)

        def test_intersect(self):
            a = self.make_rand_df(30, b=(int, 10), c=(str, 10))
            b = self.make_rand_df(80, b=(int, 50), c=(str, 50))
            c = self.make_rand_df(100, b=(int, 50), c=(str, 50))
            self.eq_sqlite('\n                SELECT * FROM c\n                    INTERSECT SELECT * FROM b\n                    INTERSECT SELECT * FROM c\n                ', a=a, b=b, c=c)

        def test_with(self):
            a = self.make_rand_df(30, a=(int, 10), b=(str, 10))
            b = self.make_rand_df(80, ax=(int, 10), bx=(str, 10))
            self.eq_sqlite('\n                WITH\n                    aa AS (\n                        SELECT a AS aa, b AS bb FROM a\n                    ),\n                    c AS (\n                        SELECT aa-1 AS aa, bb FROM aa\n                    )\n                SELECT * FROM c UNION SELECT * FROM b\n                ', a=a, b=b)

        def test_cast(self):
            pass

        def test_join_group_having(self):
            pass

        def test_integration_1(self):
            a = self.make_rand_df(100, a=int, b=str, c=float, d=int, e=bool, f=str, g=str, h=float)
            self.eq_sqlite('\n                WITH\n                    a1 AS (\n                        SELECT a+1 AS a, b, c FROM a\n                    ),\n                    a2 AS (\n                        SELECT a,MAX(b) AS b_max, AVG(c) AS c_avg FROM a GROUP BY a\n                    ),\n                    a3 AS (\n                        SELECT d+2 AS d, f, g, h FROM a WHERE e\n                    )\n                SELECT a1.a,b,c,b_max,c_avg,f,g,h FROM a1\n                    INNER JOIN a2 ON a1.a=a2.a\n                    LEFT JOIN a3 ON a1.a=a3.d\n                ', a=a)