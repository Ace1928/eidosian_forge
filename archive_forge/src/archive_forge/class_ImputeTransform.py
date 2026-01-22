from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class ImputeTransform(Transform):
    """ImputeTransform schema wrapper

    Parameters
    ----------

    impute : str, :class:`FieldName`
        The data field for which the missing values should be imputed.
    key : str, :class:`FieldName`
        A key field that uniquely identifies data objects within a group. Missing key values
        (those occurring in the data but not in the current group) will be imputed.
    frame : Sequence[None, float]
        A frame specification as a two-element array used to control the window over which
        the specified method is applied. The array entries should either be a number
        indicating the offset from the current data object, or null to indicate unbounded
        rows preceding or following the current data object. For example, the value ``[-5,
        5]`` indicates that the window should include five objects preceding and five
        objects following the current object.

        **Default value:** :  ``[null, null]`` indicating that the window includes all
        objects.
    groupby : Sequence[str, :class:`FieldName`]
        An optional array of fields by which to group the values. Imputation will then be
        performed on a per-group basis.
    keyvals : dict, Sequence[Any], :class:`ImputeSequence`
        Defines the key values that should be considered for imputation. An array of key
        values or an object defining a `number sequence
        <https://vega.github.io/vega-lite/docs/impute.html#sequence-def>`__.

        If provided, this will be used in addition to the key values observed within the
        input data. If not provided, the values will be derived from all unique values of
        the ``key`` field. For ``impute`` in ``encoding``, the key field is the x-field if
        the y-field is imputed, or vice versa.

        If there is no impute grouping, this property *must* be specified.
    method : :class:`ImputeMethod`, Literal['value', 'median', 'max', 'min', 'mean']
        The imputation method to use for the field value of imputed data objects. One of
        ``"value"``, ``"mean"``, ``"median"``, ``"max"`` or ``"min"``.

        **Default value:**  ``"value"``
    value : Any
        The field value to use when the imputation ``method`` is ``"value"``.
    """
    _schema = {'$ref': '#/definitions/ImputeTransform'}

    def __init__(self, impute: Union[str, 'SchemaBase', UndefinedType]=Undefined, key: Union[str, 'SchemaBase', UndefinedType]=Undefined, frame: Union[Sequence[Union[None, float]], UndefinedType]=Undefined, groupby: Union[Sequence[Union[str, 'SchemaBase']], UndefinedType]=Undefined, keyvals: Union[dict, 'SchemaBase', Sequence[Any], UndefinedType]=Undefined, method: Union['SchemaBase', Literal['value', 'median', 'max', 'min', 'mean'], UndefinedType]=Undefined, value: Union[Any, UndefinedType]=Undefined, **kwds):
        super(ImputeTransform, self).__init__(impute=impute, key=key, frame=frame, groupby=groupby, keyvals=keyvals, method=method, value=value, **kwds)