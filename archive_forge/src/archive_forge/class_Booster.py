import logging
from typing import Dict, Optional
import xgboost as xgb
import modin.pandas as pd
from modin.config import Engine
from modin.distributed.dataframe.pandas import unwrap_partitions
class Booster(xgb.Booster):
    """
    A Modin Booster of XGBoost.

    Booster is the model of XGBoost, that contains low level routines for
    training, prediction and evaluation.

    Parameters
    ----------
    params : dict, optional
        Parameters for boosters.
    cache : list, default: empty
        List of cache items.
    model_file : string/os.PathLike/xgb.Booster/bytearray, optional
        Path to the model file if it's string or PathLike or xgb.Booster.
    """

    def __init__(self, params=None, cache=(), model_file=None):
        super(Booster, self).__init__(params=params, cache=cache, model_file=model_file)

    def predict(self, data: DMatrix, **kwargs):
        """
        Run distributed prediction with a trained booster.

        During execution it runs ``xgb.predict`` on each worker for subset of `data`
        and creates Modin DataFrame with prediction results.

        Parameters
        ----------
        data : modin.experimental.xgboost.DMatrix
            Input data used for prediction.
        **kwargs : dict
            Other parameters are the same as for ``xgboost.Booster.predict``.

        Returns
        -------
        modin.pandas.DataFrame
            Modin DataFrame with prediction results.
        """
        LOGGER.info('Prediction started')
        if Engine.get() == 'Ray':
            from .xgboost_ray import _predict
        else:
            raise ValueError('Current version supports only Ray engine.')
        assert isinstance(data, DMatrix), f'Type of `data` is {type(data)}, but expected {DMatrix}.'
        if self.feature_names is not None and data.feature_names is not None and (self.feature_names != data.feature_names):
            data_missing = set(self.feature_names) - set(data.feature_names)
            self_missing = set(data.feature_names) - set(self.feature_names)
            msg = 'feature_names mismatch: {0} {1}'
            if data_missing:
                msg += '\nexpected ' + ', '.join((str(s) for s in data_missing)) + ' in input data'
            if self_missing:
                msg += '\ntraining data did not have the following fields: ' + ', '.join((str(s) for s in self_missing))
            raise ValueError(msg.format(self.feature_names, data.feature_names))
        result = _predict(self.copy(), data, **kwargs)
        LOGGER.info('Prediction finished')
        return result