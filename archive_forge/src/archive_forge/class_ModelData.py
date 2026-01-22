from __future__ import annotations
from statsmodels.compat.python import lmap
from functools import reduce
import numpy as np
from pandas import DataFrame, Series, isnull, MultiIndex
import statsmodels.tools.data as data_util
from statsmodels.tools.decorators import cache_readonly, cache_writable
from statsmodels.tools.sm_exceptions import MissingDataError
class ModelData:
    """
    Class responsible for handling input data and extracting metadata into the
    appropriate form
    """
    _param_names = None
    _cov_names = None

    def __init__(self, endog, exog=None, missing='none', hasconst=None, **kwargs):
        if data_util._is_recarray(endog) or data_util._is_recarray(exog):
            from statsmodels.tools.sm_exceptions import recarray_exception
            raise NotImplementedError(recarray_exception)
        if 'design_info' in kwargs:
            self.design_info = kwargs.pop('design_info')
        if 'formula' in kwargs:
            self.formula = kwargs.pop('formula')
        if missing != 'none':
            arrays, nan_idx = self.handle_missing(endog, exog, missing, **kwargs)
            self.missing_row_idx = nan_idx
            self.__dict__.update(arrays)
            self.orig_endog = self.endog
            self.orig_exog = self.exog
            self.endog, self.exog = self._convert_endog_exog(self.endog, self.exog)
        else:
            self.__dict__.update(kwargs)
            self.orig_endog = endog
            self.orig_exog = exog
            self.endog, self.exog = self._convert_endog_exog(endog, exog)
        self.const_idx = None
        self.k_constant = 0
        self._handle_constant(hasconst)
        self._check_integrity()
        self._cache = {}

    def __getstate__(self):
        from copy import copy
        d = copy(self.__dict__)
        if 'design_info' in d:
            del d['design_info']
            d['restore_design_info'] = True
        return d

    def __setstate__(self, d):
        if 'restore_design_info' in d:
            from patsy import dmatrices, PatsyError
            exc = []
            try:
                data = d['frame']
            except KeyError:
                data = d['orig_endog'].join(d['orig_exog'])
            for depth in [2, 3, 1, 0, 4]:
                try:
                    _, design = dmatrices(d['formula'], data, eval_env=depth, return_type='dataframe')
                    break
                except (NameError, PatsyError) as e:
                    exc.append(e)
                    pass
            else:
                raise exc[-1]
            self.design_info = design.design_info
            del d['restore_design_info']
        self.__dict__.update(d)

    def _handle_constant(self, hasconst):
        if hasconst is False or self.exog is None:
            self.k_constant = 0
            self.const_idx = None
        else:
            check_implicit = False
            exog_max = np.max(self.exog, axis=0)
            if not np.isfinite(exog_max).all():
                raise MissingDataError('exog contains inf or nans')
            exog_min = np.min(self.exog, axis=0)
            const_idx = np.where(exog_max == exog_min)[0].squeeze()
            self.k_constant = const_idx.size
            if self.k_constant == 1:
                if self.exog[:, const_idx].mean() != 0:
                    self.const_idx = int(const_idx)
                else:
                    check_implicit = True
            elif self.k_constant > 1:
                values = []
                for idx in const_idx:
                    value = self.exog[:, idx].mean()
                    if value == 1:
                        self.k_constant = 1
                        self.const_idx = int(idx)
                        break
                    values.append(value)
                else:
                    pos = np.array(values) != 0
                    if pos.any():
                        self.k_constant = 1
                        self.const_idx = int(const_idx[pos.argmax()])
                    else:
                        check_implicit = True
            elif self.k_constant == 0:
                check_implicit = True
            else:
                pass
            if check_implicit and (not hasconst):
                augmented_exog = np.column_stack((np.ones(self.exog.shape[0]), self.exog))
                rank_augm = np.linalg.matrix_rank(augmented_exog)
                rank_orig = np.linalg.matrix_rank(self.exog)
                self.k_constant = int(rank_orig == rank_augm)
                self.const_idx = None
            elif hasconst:
                self.k_constant = 1

    @classmethod
    def _drop_nans(cls, x, nan_mask):
        return x[nan_mask]

    @classmethod
    def _drop_nans_2d(cls, x, nan_mask):
        return x[nan_mask][:, nan_mask]

    @classmethod
    def handle_missing(cls, endog, exog, missing, **kwargs):
        """
        This returns a dictionary with keys endog, exog and the keys of
        kwargs. It preserves Nones.
        """
        none_array_names = []
        missing_idx = kwargs.pop('missing_idx', None)
        if missing_idx is not None:
            combined = ()
            combined_names = []
            if exog is None:
                none_array_names += ['exog']
        elif exog is not None:
            combined = (endog, exog)
            combined_names = ['endog', 'exog']
        else:
            combined = (endog,)
            combined_names = ['endog']
            none_array_names += ['exog']
        combined_2d = ()
        combined_2d_names = []
        if len(kwargs):
            for key, value_array in kwargs.items():
                if value_array is None or np.ndim(value_array) == 0:
                    none_array_names += [key]
                    continue
                if value_array.ndim == 1:
                    combined += (np.asarray(value_array),)
                    combined_names += [key]
                elif value_array.squeeze().ndim == 1:
                    combined += (np.asarray(value_array),)
                    combined_names += [key]
                elif value_array.ndim == 2:
                    combined_2d += (np.asarray(value_array),)
                    combined_2d_names += [key]
                else:
                    raise ValueError('Arrays with more than 2 dimensions are not yet handled')
        if missing_idx is not None:
            nan_mask = missing_idx
            updated_row_mask = None
            if combined:
                combined_nans = _nan_rows(*combined)
                if combined_nans.shape[0] != nan_mask.shape[0]:
                    raise ValueError('Shape mismatch between endog/exog and extra arrays given to model.')
                updated_row_mask = combined_nans[~nan_mask]
                nan_mask |= combined_nans
            if combined_2d:
                combined_2d_nans = _nan_rows(combined_2d)
                if combined_2d_nans.shape[0] != nan_mask.shape[0]:
                    raise ValueError('Shape mismatch between endog/exog and extra 2d arrays given to model.')
                if updated_row_mask is not None:
                    updated_row_mask |= combined_2d_nans[~nan_mask]
                else:
                    updated_row_mask = combined_2d_nans[~nan_mask]
                nan_mask |= combined_2d_nans
        else:
            nan_mask = _nan_rows(*combined)
            if combined_2d:
                nan_mask = _nan_rows(*(nan_mask[:, None],) + combined_2d)
        if not np.any(nan_mask):
            combined = dict(zip(combined_names, combined))
            if combined_2d:
                combined.update(dict(zip(combined_2d_names, combined_2d)))
            if none_array_names:
                combined.update({k: kwargs.get(k, None) for k in none_array_names})
            if missing_idx is not None:
                combined.update({'endog': endog})
                if exog is not None:
                    combined.update({'exog': exog})
            return (combined, [])
        elif missing == 'raise':
            raise MissingDataError('NaNs were encountered in the data')
        elif missing == 'drop':
            nan_mask = ~nan_mask
            drop_nans = lambda x: cls._drop_nans(x, nan_mask)
            drop_nans_2d = lambda x: cls._drop_nans_2d(x, nan_mask)
            combined = dict(zip(combined_names, lmap(drop_nans, combined)))
            if missing_idx is not None:
                if updated_row_mask is not None:
                    updated_row_mask = ~updated_row_mask
                    endog = cls._drop_nans(endog, updated_row_mask)
                    if exog is not None:
                        exog = cls._drop_nans(exog, updated_row_mask)
                combined.update({'endog': endog})
                if exog is not None:
                    combined.update({'exog': exog})
            if combined_2d:
                combined.update(dict(zip(combined_2d_names, lmap(drop_nans_2d, combined_2d))))
            if none_array_names:
                combined.update({k: kwargs.get(k, None) for k in none_array_names})
            return (combined, np.where(~nan_mask)[0].tolist())
        else:
            raise ValueError('missing option %s not understood' % missing)

    def _convert_endog_exog(self, endog, exog):
        yarr = self._get_yarr(endog)
        xarr = None
        if exog is not None:
            xarr = self._get_xarr(exog)
            if xarr.ndim == 1:
                xarr = xarr[:, None]
            if xarr.ndim != 2:
                raise ValueError('exog is not 1d or 2d')
        return (yarr, xarr)

    @cache_writable()
    def ynames(self):
        endog = self.orig_endog
        ynames = self._get_names(endog)
        if not ynames:
            ynames = _make_endog_names(self.endog)
        if len(ynames) == 1:
            return ynames[0]
        else:
            return list(ynames)

    @cache_writable()
    def xnames(self) -> list[str] | None:
        exog = self.orig_exog
        if exog is not None:
            xnames = self._get_names(exog)
            if not xnames:
                xnames = _make_exog_names(self.exog)
            return list(xnames)
        return None

    @property
    def param_names(self):
        return self._param_names or self.xnames

    @param_names.setter
    def param_names(self, values):
        self._param_names = values

    @property
    def cov_names(self):
        """
        Labels for covariance matrices

        In multidimensional models, each dimension of a covariance matrix
        differs from the number of param_names.

        If not set, returns param_names
        """
        if self._cov_names is not None:
            return self._cov_names
        return self.param_names

    @cov_names.setter
    def cov_names(self, value):
        self._cov_names = value

    @cache_readonly
    def row_labels(self):
        exog = self.orig_exog
        if exog is not None:
            row_labels = self._get_row_labels(exog)
        else:
            endog = self.orig_endog
            row_labels = self._get_row_labels(endog)
        return row_labels

    def _get_row_labels(self, arr):
        return None

    def _get_names(self, arr):
        if isinstance(arr, DataFrame):
            if isinstance(arr.columns, MultiIndex):
                return ['_'.join((level for level in c if level)) for c in arr.columns]
            else:
                return list(arr.columns)
        elif isinstance(arr, Series):
            if arr.name:
                return [arr.name]
            else:
                return
        else:
            try:
                return arr.dtype.names
            except AttributeError:
                pass
        return None

    def _get_yarr(self, endog):
        if data_util._is_structured_ndarray(endog):
            endog = data_util.struct_to_ndarray(endog)
        endog = np.asarray(endog)
        if len(endog) == 1:
            if endog.ndim == 1:
                return endog
            elif endog.ndim > 1:
                return np.asarray([endog.squeeze()])
        return endog.squeeze()

    def _get_xarr(self, exog):
        if data_util._is_structured_ndarray(exog):
            exog = data_util.struct_to_ndarray(exog)
        return np.asarray(exog)

    def _check_integrity(self):
        if self.exog is not None:
            if len(self.exog) != len(self.endog):
                raise ValueError('endog and exog matrices are different sizes')

    def wrap_output(self, obj, how='columns', names=None):
        if how == 'columns':
            return self.attach_columns(obj)
        elif how == 'rows':
            return self.attach_rows(obj)
        elif how == 'cov':
            return self.attach_cov(obj)
        elif how == 'dates':
            return self.attach_dates(obj)
        elif how == 'columns_eq':
            return self.attach_columns_eq(obj)
        elif how == 'cov_eq':
            return self.attach_cov_eq(obj)
        elif how == 'generic_columns':
            return self.attach_generic_columns(obj, names)
        elif how == 'generic_columns_2d':
            return self.attach_generic_columns_2d(obj, names)
        elif how == 'ynames':
            return self.attach_ynames(obj)
        elif how == 'multivariate_confint':
            return self.attach_mv_confint(obj)
        else:
            return obj

    def attach_columns(self, result):
        return result

    def attach_columns_eq(self, result):
        return result

    def attach_cov(self, result):
        return result

    def attach_cov_eq(self, result):
        return result

    def attach_rows(self, result):
        return result

    def attach_dates(self, result):
        return result

    def attach_mv_confint(self, result):
        return result

    def attach_generic_columns(self, result, *args, **kwargs):
        return result

    def attach_generic_columns_2d(self, result, *args, **kwargs):
        return result

    def attach_ynames(self, result):
        return result