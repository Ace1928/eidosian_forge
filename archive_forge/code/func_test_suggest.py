from fugue_dask import DaskExecutionEngine
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Lasso, LinearRegression
from tune import TUNE_OBJECT_FACTORY, Grid, Rand
from tune_hyperopt.optimizer import HyperoptLocalOptimizer
from tune_sklearn import sk_space, suggest_sk_models, suggest_sk_models_by_cv
def test_suggest(tmpdir):
    TUNE_OBJECT_FACTORY.set_temp_path(str(tmpdir))
    data = load_diabetes(as_frame=True)
    alldata = data['data']
    alldata['label'] = data['target']
    alldata = alldata.sample(frac=1, random_state=0)
    test = alldata.iloc[:20]
    train = alldata.iloc[20:]
    space1 = sk_space(LinearRegression, fit_intercept=Grid(True, False))
    space2 = sk_space(Lasso, fit_intercept=Grid(True, False), alpha=Rand(0.5, 1)).sample(3, 0)
    space3 = sk_space(Lasso, fit_intercept=Grid(True, False), alpha=Rand(0.5, 1))
    result = suggest_sk_models(space1 + space3, train, test, 'neg_mean_absolute_error', top_n=0, distributed=False, local_optimizer=HyperoptLocalOptimizer(max_iter=10, seed=0))
    assert 4 == len(result)
    assert 50 > result[0].sort_metric
    result = suggest_sk_models(space1 + space2, train, test, 'neg_mean_absolute_error', top_n=0, partition_keys=['sex'], temp_path=str(tmpdir), save_model=True)
    assert 16 == len(result)
    assert 50 > result[0].sort_metric
    result = suggest_sk_models(space1 + space2, train, test, 'neg_mean_absolute_error', top_n=1, partition_keys=['sex'], execution_engine=DaskExecutionEngine)
    assert 2 == len(result)
    assert 50 > result[0].sort_metric