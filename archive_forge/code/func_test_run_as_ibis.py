from unittest import TestCase
import ibis
from fugue import ExecutionEngine, FugueWorkflow, register_default_sql_engine
from fugue_ibis import IbisEngine, as_fugue, as_ibis, run_ibis
def test_run_as_ibis(self):
    dag = FugueWorkflow()
    df = dag.df([[0, 1], [2, 3]], 'a:long,b:long')
    idf = as_ibis(df)
    res = as_fugue(idf)
    res.assert_eq(df)
    dag.run(self.engine)
    dag = FugueWorkflow()
    df1 = dag.df([[0, 1], [2, 3]], 'a:long,b:long')
    df2 = dag.df([[0, ['x']], [3, ['y']]], 'a:long,c:[str]')
    idf1 = as_ibis(df1)
    idf2 = as_ibis(df2)
    idf = idf1.inner_join(idf2, idf1.a == idf2.a)[idf1, idf2.c]
    res = as_fugue(idf)
    expected = dag.df([[0, 1, ['x']]], 'a:long,b:long,c:[str]')
    res.assert_eq(expected, check_order=True, check_schema=True)
    dag.run(self.engine)
    dag = FugueWorkflow()
    idf1 = dag.df([[0, 1], [2, 3]], 'a:long,b:long').as_ibis()
    idf2 = dag.df([[0, ['x']], [3, ['y']]], 'a:long,c:[str]').as_ibis()
    res = idf1.inner_join(idf2, idf1.a == idf2.a)[idf1, idf2.c].as_fugue()
    expected = dag.df([[0, 1, ['x']]], 'a:long,b:long,c:[str]')
    res.assert_eq(expected, check_order=True, check_schema=True)
    dag.run(self.engine)