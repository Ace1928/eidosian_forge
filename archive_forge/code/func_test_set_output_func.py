import warnings
import numpy as np
import pytest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.preprocessing._function_transformer import _get_adapter_from_container
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS
def test_set_output_func():
    """Check behavior of set_output with different settings."""
    pd = pytest.importorskip('pandas')
    X = pd.DataFrame({'a': [1, 2, 3], 'b': [10, 20, 100]})
    ft = FunctionTransformer(np.log, feature_names_out='one-to-one')
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        ft.set_output(transform='pandas')
    X_trans = ft.fit_transform(X)
    assert isinstance(X_trans, pd.DataFrame)
    assert_array_equal(X_trans.columns, ['a', 'b'])
    ft = FunctionTransformer(lambda x: 2 * x)
    ft.set_output(transform='pandas')
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        X_trans = ft.fit_transform(X)
    assert isinstance(X_trans, pd.DataFrame)
    assert_array_equal(X_trans.columns, ['a', 'b'])
    ft_np = FunctionTransformer(lambda x: np.asarray(x))
    for transform in ('pandas', 'polars'):
        ft_np.set_output(transform=transform)
        msg = f"When `set_output` is configured to be '{transform}'.*{transform} DataFrame.*"
        with pytest.warns(UserWarning, match=msg):
            ft_np.fit_transform(X)
    ft_np.set_output(transform='default')
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        ft_np.fit_transform(X)