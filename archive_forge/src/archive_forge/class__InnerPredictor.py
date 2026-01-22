import abc
import ctypes
import inspect
import json
import warnings
from collections import OrderedDict
from copy import deepcopy
from enum import Enum
from functools import wraps
from os import SEEK_END, environ
from os.path import getsize
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union
import numpy as np
import scipy.sparse
from .compat import (PANDAS_INSTALLED, PYARROW_INSTALLED, arrow_cffi, arrow_is_floating, arrow_is_integer, concat,
from .libpath import find_lib_path
class _InnerPredictor:
    """_InnerPredictor of LightGBM.

    Not exposed to user.
    Used only for prediction, usually used for continued training.

    .. note::

        Can be converted from Booster, but cannot be converted to Booster.
    """

    def __init__(self, booster_handle: _BoosterHandle, pandas_categorical: Optional[List[List]], pred_parameter: Dict[str, Any], manage_handle: bool):
        """Initialize the _InnerPredictor.

        Parameters
        ----------
        booster_handle : object
            Handle of Booster.
        pandas_categorical : list of list, or None
            If provided, list of categories for ``pandas`` categorical columns.
            Where the ``i``th element of the list contains the categories for the ``i``th categorical feature.
        pred_parameter : dict
            Other parameters for the prediction.
        manage_handle : bool
            If ``True``, free the corresponding Booster on the C++ side when this Python object is deleted.
        """
        self._handle = booster_handle
        self.__is_manage_handle = manage_handle
        self.pandas_categorical = pandas_categorical
        self.pred_parameter = _param_dict_to_str(pred_parameter)
        out_num_class = ctypes.c_int(0)
        _safe_call(_LIB.LGBM_BoosterGetNumClasses(self._handle, ctypes.byref(out_num_class)))
        self.num_class = out_num_class.value

    @classmethod
    def from_booster(cls, booster: 'Booster', pred_parameter: Dict[str, Any]) -> '_InnerPredictor':
        """Initialize an ``_InnerPredictor`` from a ``Booster``.

        Parameters
        ----------
        booster : Booster
            Booster.
        pred_parameter : dict
            Other parameters for the prediction.
        """
        out_cur_iter = ctypes.c_int(0)
        _safe_call(_LIB.LGBM_BoosterGetCurrentIteration(booster._handle, ctypes.byref(out_cur_iter)))
        return cls(booster_handle=booster._handle, pandas_categorical=booster.pandas_categorical, pred_parameter=pred_parameter, manage_handle=False)

    @classmethod
    def from_model_file(cls, model_file: Union[str, Path], pred_parameter: Dict[str, Any]) -> '_InnerPredictor':
        """Initialize an ``_InnerPredictor`` from a text file containing a LightGBM model.

        Parameters
        ----------
        model_file : str or pathlib.Path
            Path to the model file.
        pred_parameter : dict
            Other parameters for the prediction.
        """
        booster_handle = ctypes.c_void_p()
        out_num_iterations = ctypes.c_int(0)
        _safe_call(_LIB.LGBM_BoosterCreateFromModelfile(_c_str(str(model_file)), ctypes.byref(out_num_iterations), ctypes.byref(booster_handle)))
        return cls(booster_handle=booster_handle, pandas_categorical=_load_pandas_categorical(file_name=model_file), pred_parameter=pred_parameter, manage_handle=True)

    def __del__(self) -> None:
        try:
            if self.__is_manage_handle:
                _safe_call(_LIB.LGBM_BoosterFree(self._handle))
        except AttributeError:
            pass

    def __getstate__(self) -> Dict[str, Any]:
        this = self.__dict__.copy()
        this.pop('handle', None)
        this.pop('_handle', None)
        return this

    def predict(self, data: _LGBM_PredictDataType, start_iteration: int=0, num_iteration: int=-1, raw_score: bool=False, pred_leaf: bool=False, pred_contrib: bool=False, data_has_header: bool=False, validate_features: bool=False) -> Union[np.ndarray, scipy.sparse.spmatrix, List[scipy.sparse.spmatrix]]:
        """Predict logic.

        Parameters
        ----------
        data : str, pathlib.Path, numpy array, pandas DataFrame, pyarrow Table, H2O DataTable's Frame or scipy.sparse
            Data source for prediction.
            If str or pathlib.Path, it represents the path to a text file (CSV, TSV, or LibSVM).
        start_iteration : int, optional (default=0)
            Start index of the iteration to predict.
        num_iteration : int, optional (default=-1)
            Iteration used for prediction.
        raw_score : bool, optional (default=False)
            Whether to predict raw scores.
        pred_leaf : bool, optional (default=False)
            Whether to predict leaf index.
        pred_contrib : bool, optional (default=False)
            Whether to predict feature contributions.
        data_has_header : bool, optional (default=False)
            Whether data has header.
            Used only for txt data.
        validate_features : bool, optional (default=False)
            If True, ensure that the features used to predict match the ones used to train.
            Used only if data is pandas DataFrame.

            .. versionadded:: 4.0.0

        Returns
        -------
        result : numpy array, scipy.sparse or list of scipy.sparse
            Prediction result.
            Can be sparse or a list of sparse objects (each element represents predictions for one class) for feature contributions (when ``pred_contrib=True``).
        """
        if isinstance(data, Dataset):
            raise TypeError('Cannot use Dataset instance for prediction, please use raw data instead')
        elif isinstance(data, pd_DataFrame) and validate_features:
            data_names = [str(x) for x in data.columns]
            ptr_names = (ctypes.c_char_p * len(data_names))()
            ptr_names[:] = [x.encode('utf-8') for x in data_names]
            _safe_call(_LIB.LGBM_BoosterValidateFeatureNames(self._handle, ptr_names, ctypes.c_int(len(data_names))))
        if isinstance(data, pd_DataFrame):
            data = _data_from_pandas(data=data, feature_name='auto', categorical_feature='auto', pandas_categorical=self.pandas_categorical)[0]
        predict_type = _C_API_PREDICT_NORMAL
        if raw_score:
            predict_type = _C_API_PREDICT_RAW_SCORE
        if pred_leaf:
            predict_type = _C_API_PREDICT_LEAF_INDEX
        if pred_contrib:
            predict_type = _C_API_PREDICT_CONTRIB
        if isinstance(data, (str, Path)):
            with _TempFile() as f:
                _safe_call(_LIB.LGBM_BoosterPredictForFile(self._handle, _c_str(str(data)), ctypes.c_int(data_has_header), ctypes.c_int(predict_type), ctypes.c_int(start_iteration), ctypes.c_int(num_iteration), _c_str(self.pred_parameter), _c_str(f.name)))
                preds = np.loadtxt(f.name, dtype=np.float64)
                nrow = preds.shape[0]
        elif isinstance(data, scipy.sparse.csr_matrix):
            preds, nrow = self.__pred_for_csr(csr=data, start_iteration=start_iteration, num_iteration=num_iteration, predict_type=predict_type)
        elif isinstance(data, scipy.sparse.csc_matrix):
            preds, nrow = self.__pred_for_csc(csc=data, start_iteration=start_iteration, num_iteration=num_iteration, predict_type=predict_type)
        elif isinstance(data, np.ndarray):
            preds, nrow = self.__pred_for_np2d(mat=data, start_iteration=start_iteration, num_iteration=num_iteration, predict_type=predict_type)
        elif _is_pyarrow_table(data):
            preds, nrow = self.__pred_for_pyarrow_table(table=data, start_iteration=start_iteration, num_iteration=num_iteration, predict_type=predict_type)
        elif isinstance(data, list):
            try:
                data = np.array(data)
            except BaseException as err:
                raise ValueError('Cannot convert data list to numpy array.') from err
            preds, nrow = self.__pred_for_np2d(mat=data, start_iteration=start_iteration, num_iteration=num_iteration, predict_type=predict_type)
        elif isinstance(data, dt_DataTable):
            preds, nrow = self.__pred_for_np2d(mat=data.to_numpy(), start_iteration=start_iteration, num_iteration=num_iteration, predict_type=predict_type)
        else:
            try:
                _log_warning('Converting data to scipy sparse matrix.')
                csr = scipy.sparse.csr_matrix(data)
            except BaseException as err:
                raise TypeError(f'Cannot predict data for type {type(data).__name__}') from err
            preds, nrow = self.__pred_for_csr(csr=csr, start_iteration=start_iteration, num_iteration=num_iteration, predict_type=predict_type)
        if pred_leaf:
            preds = preds.astype(np.int32)
        is_sparse = isinstance(preds, scipy.sparse.spmatrix) or isinstance(preds, list)
        if not is_sparse and preds.size != nrow:
            if preds.size % nrow == 0:
                preds = preds.reshape(nrow, -1)
            else:
                raise ValueError(f'Length of predict result ({preds.size}) cannot be divide nrow ({nrow})')
        return preds

    def __get_num_preds(self, start_iteration: int, num_iteration: int, nrow: int, predict_type: int) -> int:
        """Get size of prediction result."""
        if nrow > _MAX_INT32:
            raise LightGBMError(f'LightGBM cannot perform prediction for data with number of rows greater than MAX_INT32 ({_MAX_INT32}).\nYou can split your data into chunks and then concatenate predictions for them')
        n_preds = ctypes.c_int64(0)
        _safe_call(_LIB.LGBM_BoosterCalcNumPredict(self._handle, ctypes.c_int(nrow), ctypes.c_int(predict_type), ctypes.c_int(start_iteration), ctypes.c_int(num_iteration), ctypes.byref(n_preds)))
        return n_preds.value

    def __inner_predict_np2d(self, mat: np.ndarray, start_iteration: int, num_iteration: int, predict_type: int, preds: Optional[np.ndarray]) -> Tuple[np.ndarray, int]:
        if mat.dtype == np.float32 or mat.dtype == np.float64:
            data = np.array(mat.reshape(mat.size), dtype=mat.dtype, copy=False)
        else:
            data = np.array(mat.reshape(mat.size), dtype=np.float32)
        ptr_data, type_ptr_data, _ = _c_float_array(data)
        n_preds = self.__get_num_preds(start_iteration=start_iteration, num_iteration=num_iteration, nrow=mat.shape[0], predict_type=predict_type)
        if preds is None:
            preds = np.empty(n_preds, dtype=np.float64)
        elif len(preds.shape) != 1 or len(preds) != n_preds:
            raise ValueError('Wrong length of pre-allocated predict array')
        out_num_preds = ctypes.c_int64(0)
        _safe_call(_LIB.LGBM_BoosterPredictForMat(self._handle, ptr_data, ctypes.c_int(type_ptr_data), ctypes.c_int32(mat.shape[0]), ctypes.c_int32(mat.shape[1]), ctypes.c_int(_C_API_IS_ROW_MAJOR), ctypes.c_int(predict_type), ctypes.c_int(start_iteration), ctypes.c_int(num_iteration), _c_str(self.pred_parameter), ctypes.byref(out_num_preds), preds.ctypes.data_as(ctypes.POINTER(ctypes.c_double))))
        if n_preds != out_num_preds.value:
            raise ValueError('Wrong length for predict results')
        return (preds, mat.shape[0])

    def __pred_for_np2d(self, mat: np.ndarray, start_iteration: int, num_iteration: int, predict_type: int) -> Tuple[np.ndarray, int]:
        """Predict for a 2-D numpy matrix."""
        if len(mat.shape) != 2:
            raise ValueError('Input numpy.ndarray or list must be 2 dimensional')
        nrow = mat.shape[0]
        if nrow > _MAX_INT32:
            sections = np.arange(start=_MAX_INT32, stop=nrow, step=_MAX_INT32)
            n_preds = [self.__get_num_preds(start_iteration, num_iteration, i, predict_type) for i in np.diff([0] + list(sections) + [nrow])]
            n_preds_sections = np.array([0] + n_preds, dtype=np.intp).cumsum()
            preds = np.empty(sum(n_preds), dtype=np.float64)
            for chunk, (start_idx_pred, end_idx_pred) in zip(np.array_split(mat, sections), zip(n_preds_sections, n_preds_sections[1:])):
                self.__inner_predict_np2d(mat=chunk, start_iteration=start_iteration, num_iteration=num_iteration, predict_type=predict_type, preds=preds[start_idx_pred:end_idx_pred])
            return (preds, nrow)
        else:
            return self.__inner_predict_np2d(mat=mat, start_iteration=start_iteration, num_iteration=num_iteration, predict_type=predict_type, preds=None)

    def __create_sparse_native(self, cs: Union[scipy.sparse.csc_matrix, scipy.sparse.csr_matrix], out_shape: np.ndarray, out_ptr_indptr: 'ctypes._Pointer', out_ptr_indices: 'ctypes._Pointer', out_ptr_data: 'ctypes._Pointer', indptr_type: int, data_type: int, is_csr: bool) -> Union[List[scipy.sparse.csc_matrix], List[scipy.sparse.csr_matrix]]:
        data_indices_len = out_shape[0]
        indptr_len = out_shape[1]
        if indptr_type == _C_API_DTYPE_INT32:
            out_indptr = _cint32_array_to_numpy(cptr=out_ptr_indptr, length=indptr_len)
        elif indptr_type == _C_API_DTYPE_INT64:
            out_indptr = _cint64_array_to_numpy(cptr=out_ptr_indptr, length=indptr_len)
        else:
            raise TypeError('Expected int32 or int64 type for indptr')
        if data_type == _C_API_DTYPE_FLOAT32:
            out_data = _cfloat32_array_to_numpy(cptr=out_ptr_data, length=data_indices_len)
        elif data_type == _C_API_DTYPE_FLOAT64:
            out_data = _cfloat64_array_to_numpy(cptr=out_ptr_data, length=data_indices_len)
        else:
            raise TypeError('Expected float32 or float64 type for data')
        out_indices = _cint32_array_to_numpy(cptr=out_ptr_indices, length=data_indices_len)
        per_class_indptr_shape = cs.indptr.shape[0]
        if not is_csr:
            per_class_indptr_shape += 1
        out_indptr_arrays = np.split(out_indptr, out_indptr.shape[0] / per_class_indptr_shape)
        cs_output_matrices = []
        offset = 0
        for cs_indptr in out_indptr_arrays:
            matrix_indptr_len = cs_indptr[cs_indptr.shape[0] - 1]
            cs_indices = out_indices[offset + cs_indptr[0]:offset + matrix_indptr_len]
            cs_data = out_data[offset + cs_indptr[0]:offset + matrix_indptr_len]
            offset += matrix_indptr_len
            cs_shape = [cs.shape[0], cs.shape[1] + 1]
            if is_csr:
                cs_output_matrices.append(scipy.sparse.csr_matrix((cs_data, cs_indices, cs_indptr), cs_shape))
            else:
                cs_output_matrices.append(scipy.sparse.csc_matrix((cs_data, cs_indices, cs_indptr), cs_shape))
        _safe_call(_LIB.LGBM_BoosterFreePredictSparse(out_ptr_indptr, out_ptr_indices, out_ptr_data, ctypes.c_int(indptr_type), ctypes.c_int(data_type)))
        if len(cs_output_matrices) == 1:
            return cs_output_matrices[0]
        return cs_output_matrices

    def __inner_predict_csr(self, csr: scipy.sparse.csr_matrix, start_iteration: int, num_iteration: int, predict_type: int, preds: Optional[np.ndarray]) -> Tuple[np.ndarray, int]:
        nrow = len(csr.indptr) - 1
        n_preds = self.__get_num_preds(start_iteration=start_iteration, num_iteration=num_iteration, nrow=nrow, predict_type=predict_type)
        if preds is None:
            preds = np.empty(n_preds, dtype=np.float64)
        elif len(preds.shape) != 1 or len(preds) != n_preds:
            raise ValueError('Wrong length of pre-allocated predict array')
        out_num_preds = ctypes.c_int64(0)
        ptr_indptr, type_ptr_indptr, _ = _c_int_array(csr.indptr)
        ptr_data, type_ptr_data, _ = _c_float_array(csr.data)
        assert csr.shape[1] <= _MAX_INT32
        csr_indices = csr.indices.astype(np.int32, copy=False)
        _safe_call(_LIB.LGBM_BoosterPredictForCSR(self._handle, ptr_indptr, ctypes.c_int(type_ptr_indptr), csr_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), ptr_data, ctypes.c_int(type_ptr_data), ctypes.c_int64(len(csr.indptr)), ctypes.c_int64(len(csr.data)), ctypes.c_int64(csr.shape[1]), ctypes.c_int(predict_type), ctypes.c_int(start_iteration), ctypes.c_int(num_iteration), _c_str(self.pred_parameter), ctypes.byref(out_num_preds), preds.ctypes.data_as(ctypes.POINTER(ctypes.c_double))))
        if n_preds != out_num_preds.value:
            raise ValueError('Wrong length for predict results')
        return (preds, nrow)

    def __inner_predict_csr_sparse(self, csr: scipy.sparse.csr_matrix, start_iteration: int, num_iteration: int, predict_type: int) -> Tuple[Union[List[scipy.sparse.csc_matrix], List[scipy.sparse.csr_matrix]], int]:
        ptr_indptr, type_ptr_indptr, __ = _c_int_array(csr.indptr)
        ptr_data, type_ptr_data, _ = _c_float_array(csr.data)
        csr_indices = csr.indices.astype(np.int32, copy=False)
        matrix_type = _C_API_MATRIX_TYPE_CSR
        out_ptr_indptr: _ctypes_int_ptr
        if type_ptr_indptr == _C_API_DTYPE_INT32:
            out_ptr_indptr = ctypes.POINTER(ctypes.c_int32)()
        else:
            out_ptr_indptr = ctypes.POINTER(ctypes.c_int64)()
        out_ptr_indices = ctypes.POINTER(ctypes.c_int32)()
        out_ptr_data: _ctypes_float_ptr
        if type_ptr_data == _C_API_DTYPE_FLOAT32:
            out_ptr_data = ctypes.POINTER(ctypes.c_float)()
        else:
            out_ptr_data = ctypes.POINTER(ctypes.c_double)()
        out_shape = np.empty(2, dtype=np.int64)
        _safe_call(_LIB.LGBM_BoosterPredictSparseOutput(self._handle, ptr_indptr, ctypes.c_int(type_ptr_indptr), csr_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), ptr_data, ctypes.c_int(type_ptr_data), ctypes.c_int64(len(csr.indptr)), ctypes.c_int64(len(csr.data)), ctypes.c_int64(csr.shape[1]), ctypes.c_int(predict_type), ctypes.c_int(start_iteration), ctypes.c_int(num_iteration), _c_str(self.pred_parameter), ctypes.c_int(matrix_type), out_shape.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)), ctypes.byref(out_ptr_indptr), ctypes.byref(out_ptr_indices), ctypes.byref(out_ptr_data)))
        matrices = self.__create_sparse_native(cs=csr, out_shape=out_shape, out_ptr_indptr=out_ptr_indptr, out_ptr_indices=out_ptr_indices, out_ptr_data=out_ptr_data, indptr_type=type_ptr_indptr, data_type=type_ptr_data, is_csr=True)
        nrow = len(csr.indptr) - 1
        return (matrices, nrow)

    def __pred_for_csr(self, csr: scipy.sparse.csr_matrix, start_iteration: int, num_iteration: int, predict_type: int) -> Tuple[np.ndarray, int]:
        """Predict for a CSR data."""
        if predict_type == _C_API_PREDICT_CONTRIB:
            return self.__inner_predict_csr_sparse(csr=csr, start_iteration=start_iteration, num_iteration=num_iteration, predict_type=predict_type)
        nrow = len(csr.indptr) - 1
        if nrow > _MAX_INT32:
            sections = [0] + list(np.arange(start=_MAX_INT32, stop=nrow, step=_MAX_INT32)) + [nrow]
            n_preds = [self.__get_num_preds(start_iteration, num_iteration, i, predict_type) for i in np.diff(sections)]
            n_preds_sections = np.array([0] + n_preds, dtype=np.intp).cumsum()
            preds = np.empty(sum(n_preds), dtype=np.float64)
            for (start_idx, end_idx), (start_idx_pred, end_idx_pred) in zip(zip(sections, sections[1:]), zip(n_preds_sections, n_preds_sections[1:])):
                self.__inner_predict_csr(csr=csr[start_idx:end_idx], start_iteration=start_iteration, num_iteration=num_iteration, predict_type=predict_type, preds=preds[start_idx_pred:end_idx_pred])
            return (preds, nrow)
        else:
            return self.__inner_predict_csr(csr=csr, start_iteration=start_iteration, num_iteration=num_iteration, predict_type=predict_type, preds=None)

    def __inner_predict_sparse_csc(self, csc: scipy.sparse.csc_matrix, start_iteration: int, num_iteration: int, predict_type: int):
        ptr_indptr, type_ptr_indptr, __ = _c_int_array(csc.indptr)
        ptr_data, type_ptr_data, _ = _c_float_array(csc.data)
        csc_indices = csc.indices.astype(np.int32, copy=False)
        matrix_type = _C_API_MATRIX_TYPE_CSC
        out_ptr_indptr: _ctypes_int_ptr
        if type_ptr_indptr == _C_API_DTYPE_INT32:
            out_ptr_indptr = ctypes.POINTER(ctypes.c_int32)()
        else:
            out_ptr_indptr = ctypes.POINTER(ctypes.c_int64)()
        out_ptr_indices = ctypes.POINTER(ctypes.c_int32)()
        out_ptr_data: _ctypes_float_ptr
        if type_ptr_data == _C_API_DTYPE_FLOAT32:
            out_ptr_data = ctypes.POINTER(ctypes.c_float)()
        else:
            out_ptr_data = ctypes.POINTER(ctypes.c_double)()
        out_shape = np.empty(2, dtype=np.int64)
        _safe_call(_LIB.LGBM_BoosterPredictSparseOutput(self._handle, ptr_indptr, ctypes.c_int(type_ptr_indptr), csc_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), ptr_data, ctypes.c_int(type_ptr_data), ctypes.c_int64(len(csc.indptr)), ctypes.c_int64(len(csc.data)), ctypes.c_int64(csc.shape[0]), ctypes.c_int(predict_type), ctypes.c_int(start_iteration), ctypes.c_int(num_iteration), _c_str(self.pred_parameter), ctypes.c_int(matrix_type), out_shape.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)), ctypes.byref(out_ptr_indptr), ctypes.byref(out_ptr_indices), ctypes.byref(out_ptr_data)))
        matrices = self.__create_sparse_native(cs=csc, out_shape=out_shape, out_ptr_indptr=out_ptr_indptr, out_ptr_indices=out_ptr_indices, out_ptr_data=out_ptr_data, indptr_type=type_ptr_indptr, data_type=type_ptr_data, is_csr=False)
        nrow = csc.shape[0]
        return (matrices, nrow)

    def __pred_for_csc(self, csc: scipy.sparse.csc_matrix, start_iteration: int, num_iteration: int, predict_type: int) -> Tuple[np.ndarray, int]:
        """Predict for a CSC data."""
        nrow = csc.shape[0]
        if nrow > _MAX_INT32:
            return self.__pred_for_csr(csr=csc.tocsr(), start_iteration=start_iteration, num_iteration=num_iteration, predict_type=predict_type)
        if predict_type == _C_API_PREDICT_CONTRIB:
            return self.__inner_predict_sparse_csc(csc=csc, start_iteration=start_iteration, num_iteration=num_iteration, predict_type=predict_type)
        n_preds = self.__get_num_preds(start_iteration=start_iteration, num_iteration=num_iteration, nrow=nrow, predict_type=predict_type)
        preds = np.empty(n_preds, dtype=np.float64)
        out_num_preds = ctypes.c_int64(0)
        ptr_indptr, type_ptr_indptr, __ = _c_int_array(csc.indptr)
        ptr_data, type_ptr_data, _ = _c_float_array(csc.data)
        assert csc.shape[0] <= _MAX_INT32
        csc_indices = csc.indices.astype(np.int32, copy=False)
        _safe_call(_LIB.LGBM_BoosterPredictForCSC(self._handle, ptr_indptr, ctypes.c_int(type_ptr_indptr), csc_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), ptr_data, ctypes.c_int(type_ptr_data), ctypes.c_int64(len(csc.indptr)), ctypes.c_int64(len(csc.data)), ctypes.c_int64(csc.shape[0]), ctypes.c_int(predict_type), ctypes.c_int(start_iteration), ctypes.c_int(num_iteration), _c_str(self.pred_parameter), ctypes.byref(out_num_preds), preds.ctypes.data_as(ctypes.POINTER(ctypes.c_double))))
        if n_preds != out_num_preds.value:
            raise ValueError('Wrong length for predict results')
        return (preds, nrow)

    def __pred_for_pyarrow_table(self, table: pa_Table, start_iteration: int, num_iteration: int, predict_type: int) -> Tuple[np.ndarray, int]:
        """Predict for a PyArrow table."""
        if not PYARROW_INSTALLED:
            raise LightGBMError('Cannot predict from Arrow without `pyarrow` installed.')
        if not all((arrow_is_integer(t) or arrow_is_floating(t) for t in table.schema.types)):
            raise ValueError('Arrow table may only have integer or floating point datatypes')
        n_preds = self.__get_num_preds(start_iteration=start_iteration, num_iteration=num_iteration, nrow=table.num_rows, predict_type=predict_type)
        preds = np.empty(n_preds, dtype=np.float64)
        out_num_preds = ctypes.c_int64(0)
        c_array = _export_arrow_to_c(table)
        _safe_call(_LIB.LGBM_BoosterPredictForArrow(self._handle, ctypes.c_int64(c_array.n_chunks), ctypes.c_void_p(c_array.chunks_ptr), ctypes.c_void_p(c_array.schema_ptr), ctypes.c_int(predict_type), ctypes.c_int(start_iteration), ctypes.c_int(num_iteration), _c_str(self.pred_parameter), ctypes.byref(out_num_preds), preds.ctypes.data_as(ctypes.POINTER(ctypes.c_double))))
        if n_preds != out_num_preds.value:
            raise ValueError('Wrong length for predict results')
        return (preds, table.num_rows)

    def current_iteration(self) -> int:
        """Get the index of the current iteration.

        Returns
        -------
        cur_iter : int
            The index of the current iteration.
        """
        out_cur_iter = ctypes.c_int(0)
        _safe_call(_LIB.LGBM_BoosterGetCurrentIteration(self._handle, ctypes.byref(out_cur_iter)))
        return out_cur_iter.value