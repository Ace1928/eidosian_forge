from google.protobuf import text_format
from tensorflow.core.config import flags
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import tensor_pb2
from tensorflow.core.framework import tensor_shape_pb2
from tensorflow.core.framework import types_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import op_callbacks
from tensorflow.python.framework import op_def_library_pybind
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import compat
from tensorflow.python.util import tf_contextlib
def _ExtractInputsAndAttrs(op_type_name, op_def, allowed_list_attr_map, keywords, default_type_attr_map, attrs, inputs, input_types):
    """Extracts `attrs`, `inputs`, and `input_types` in _apply_op_helper."""
    inferred_from = {}
    for input_arg in op_def.input_arg:
        input_name = input_arg.name
        if input_name in keywords:
            values = keywords.pop(input_name)
        elif input_name + '_' in keywords:
            input_name += '_'
            values = keywords.pop(input_name)
        else:
            raise TypeError(f'No argument for input {input_name} found in {op_def}')
        if _IsListParameter(input_arg):
            if not _IsListValue(values):
                raise TypeError(f"Expected list for '{input_name}' argument to '{op_type_name}' Op, not {values}.")
            dtype = None
            default_dtype = None
            if input_arg.type != types_pb2.DT_INVALID:
                dtype = input_arg.type
            elif input_arg.number_attr:
                if input_arg.type_attr in attrs:
                    dtype = attrs[input_arg.type_attr]
                else:
                    for t in values:
                        if isinstance(t, tensor.Tensor):
                            dtype = t.dtype
                            break
                if dtype is None and input_arg.type_attr in default_type_attr_map:
                    default_dtype = default_type_attr_map[input_arg.type_attr]
            try:
                if not input_arg.is_ref and dtype:
                    dtype = dtypes.as_dtype(dtype).base_dtype
                values = ops.internal_convert_n_to_tensor(values, name=input_arg.name, dtype=dtype if dtype else None, preferred_dtype=default_dtype, as_ref=input_arg.is_ref)
                all_types = set((v.dtype.base_dtype for v in values))
                if input_arg.number_attr and len(all_types) > 1:
                    raise TypeError(f'Not all types matched for {input_arg.name} for {op_type_name}. Got {all_types}')
            except (TypeError, ValueError):
                observed_types = []
                for value in values:
                    try:
                        converted_value = ops.convert_to_tensor(value, as_ref=input_arg.is_ref)
                        observed_types.append(converted_value.dtype.base_dtype.name)
                    except (TypeError, ValueError):
                        observed_types.append('<NOT CONVERTIBLE TO TENSOR>')
                observed = ', '.join(observed_types)
                prefix = "Tensors in list passed to '%s' of '%s' Op have types [%s]" % (input_name, op_type_name, observed)
                if input_arg.number_attr:
                    if input_arg.type != types_pb2.DT_INVALID:
                        raise TypeError(f'{prefix} that do not match expected type {dtype.name}.')
                    elif input_arg.type_attr in attrs:
                        raise TypeError(f'{prefix} that do not match type {dtype.name} inferred from earlier arguments.')
                    else:
                        raise TypeError(f"{prefix} that don't all match.")
                else:
                    raise TypeError(f'{prefix} that are invalid. Tensors: {values}')
            types = [x.dtype for x in values]
            inputs.extend(values)
        else:
            dtype = None
            default_dtype = None
            allowed_list = None
            if input_arg.type != types_pb2.DT_INVALID:
                dtype = input_arg.type
            elif input_arg.type_attr in attrs:
                dtype = attrs[input_arg.type_attr]
            elif input_arg.type_attr in default_type_attr_map:
                default_dtype = default_type_attr_map[input_arg.type_attr]
                allowed_list = allowed_list_attr_map.get(input_arg.type_attr)
            try:
                if dtype is None and allowed_list:
                    inferred = None
                    try:
                        inferred = ops.convert_to_tensor(values, name=input_arg.name, as_ref=input_arg.is_ref)
                    except TypeError as err:
                        pass
                    if inferred is not None and inferred.dtype in allowed_list:
                        values = inferred
                    else:
                        values = ops.convert_to_tensor(values, name=input_arg.name, as_ref=input_arg.is_ref, preferred_dtype=default_dtype)
                else:
                    values = ops.convert_to_tensor(values, name=input_arg.name, dtype=dtype, as_ref=input_arg.is_ref, preferred_dtype=default_dtype)
            except TypeError as err:
                if dtype is None:
                    raise err
                else:
                    raise TypeError(f"Expected {dtypes.as_dtype(dtype).name} passed to parameter '{input_arg.name}' of op '{op_type_name}', got {repr(values)} of type '{type(values).__name__}' instead. Error: {err}")
            except ValueError:
                try:
                    observed = ops.convert_to_tensor(values, as_ref=input_arg.is_ref).dtype.name
                except ValueError as err:
                    raise ValueError(f"Tried to convert '{input_name}' to a tensor and failed. Error: {err}")
                prefix = "Input '%s' of '%s' Op has type %s that does not match" % (input_name, op_type_name, observed)
                if input_arg.type != types_pb2.DT_INVALID:
                    raise TypeError(f'{prefix} expected type of {dtypes.as_dtype(input_arg.type).name}.')
                else:
                    k = input_arg.type_attr
                    if k in default_type_attr_map:
                        if k not in attrs:
                            attrs[k] = default_type_attr_map[k]
                            if k not in inferred_from:
                                inferred_from[k] = 'Default in OpDef'
                    raise TypeError(f"{prefix} type {dtypes.as_dtype(attrs[input_arg.type_attr]).name} of argument '{inferred_from[input_arg.type_attr]}'.")
            types = [values.dtype]
            inputs.append(values)
        base_types = [x.base_dtype for x in types]
        if input_arg.number_attr:
            if input_arg.number_attr in attrs:
                if len(values) != attrs[input_arg.number_attr]:
                    raise ValueError(f"List argument '{input_name}' to '{op_type_name}' Op with length {len(values)} must match length {attrs[input_arg.number_attr]} of argument '{inferred_from[input_arg.number_attr]}'.")
            else:
                attrs[input_arg.number_attr] = len(values)
                inferred_from[input_arg.number_attr] = input_name
                num_attr = _Attr(op_def, input_arg.number_attr)
                if num_attr.has_minimum and len(values) < num_attr.minimum:
                    raise ValueError(f"List argument '{input_name}' to '{op_type_name}' Op with length {len(values)} shorter than minimum length {num_attr.minimum}.")
            if any((bt != base_types[0] for bt in base_types)):
                raise TypeError(f"All tensors passed to '{input_name}' of '{op_type_name}' Op must have the same type. Got {base_types} instead.")
            if input_arg.type != types_pb2.DT_INVALID:
                if base_types and base_types[0] != input_arg.type:
                    assert False, 'Unreachable'
            elif input_arg.type_attr in attrs:
                if base_types and base_types[0] != attrs[input_arg.type_attr]:
                    assert False, 'Unreachable'
            elif not base_types:
                if input_arg.type_attr not in default_type_attr_map:
                    raise TypeError(f"Don't know how to infer type variable from empty input list passed to input '{input_name}' of '{op_type_name}' Op.")
            else:
                attrs[input_arg.type_attr] = base_types[0]
                inferred_from[input_arg.type_attr] = input_name
                type_attr = _Attr(op_def, input_arg.type_attr)
                _SatisfiesTypeConstraint(base_types[0], type_attr, param_name=input_name)
        elif input_arg.type_attr:
            attr_value = base_types[0]
            if input_arg.type_attr in attrs:
                if attrs[input_arg.type_attr] != attr_value:
                    raise TypeError(f"Input '{input_name}' of '{op_type_name}' Op has type {dtypes.as_dtype(attr_value).name} that does not match type {dtypes.as_dtype(attrs[input_arg.type_attr]).name} of argument '{inferred_from[input_arg.type_attr]}'.")
            else:
                for base_type in base_types:
                    _SatisfiesTypeConstraint(base_type, _Attr(op_def, input_arg.type_attr), param_name=input_name)
                attrs[input_arg.type_attr] = attr_value
                inferred_from[input_arg.type_attr] = input_name
        elif input_arg.type_list_attr:
            attr_value = base_types
            if input_arg.type_list_attr in attrs:
                if attrs[input_arg.type_list_attr] != attr_value:
                    actual_types = ', '.join((dtypes.as_dtype(x).name for x in attr_value))
                    expected_types = ', '.join((dtypes.as_dtype(x).name for x in attrs[input_arg.type_list_attr]))
                    raise TypeError(f"Input '{input_name}' of '{op_type_name}' Op has type list of {actual_types} that does not match type list {expected_types} of argument '{inferred_from[input_arg.type_list_attr]}'.")
            else:
                for base_type in base_types:
                    _SatisfiesTypeConstraint(base_type, _Attr(op_def, input_arg.type_list_attr), param_name=input_name)
                attrs[input_arg.type_list_attr] = attr_value
                inferred_from[input_arg.type_list_attr] = input_name
        elif base_types[0] != input_arg.type:
            assert False, 'Unreachable'
        if input_arg.is_ref:
            if not all((x._is_ref_dtype for x in types)):
                raise TypeError(f"'{op_type_name}' Op requires that input '{input_name}' be a mutable tensor (e.g.: a tf.Variable)")
            input_types.extend(types)
        else:
            input_types.extend(base_types)