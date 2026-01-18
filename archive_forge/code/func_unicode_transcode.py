import collections
from tensorflow.python import pywrap_tfe as pywrap_tfe
from tensorflow.python.eager import context as _context
from tensorflow.python.eager import core as _core
from tensorflow.python.eager import execute as _execute
from tensorflow.python.framework import dtypes as _dtypes
from tensorflow.security.fuzzing.py import annotation_types as _atypes
from tensorflow.python.framework import op_def_registry as _op_def_registry
from tensorflow.python.framework import ops as _ops
from tensorflow.python.framework import op_def_library as _op_def_library
from tensorflow.python.util.deprecation import deprecated_endpoints
from tensorflow.python.util import dispatch as _dispatch
from tensorflow.python.util.tf_export import tf_export
from typing import TypeVar, List
@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('strings.unicode_transcode')
def unicode_transcode(input: _atypes.TensorFuzzingAnnotation[_atypes.String], input_encoding: str, output_encoding: str, errors: str='replace', replacement_char: int=65533, replace_control_characters: bool=False, name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.String]:
    """Transcode the input text from a source encoding to a destination encoding.

  The input is a string tensor of any shape. The output is a string tensor of
  the same shape containing the transcoded strings. Output strings are always
  valid unicode. If the input contains invalid encoding positions, the
  `errors` attribute sets the policy for how to deal with them. If the default
  error-handling policy is used, invalid formatting will be substituted in the
  output by the `replacement_char`. If the errors policy is to `ignore`, any
  invalid encoding positions in the input are skipped and not included in the
  output. If it set to `strict` then any invalid formatting will result in an
  InvalidArgument error.

  This operation can be used with `output_encoding = input_encoding` to enforce
  correct formatting for inputs even if they are already in the desired encoding.

  If the input is prefixed by a Byte Order Mark needed to determine encoding
  (e.g. if the encoding is UTF-16 and the BOM indicates big-endian), then that
  BOM will be consumed and not emitted into the output. If the input encoding
  is marked with an explicit endianness (e.g. UTF-16-BE), then the BOM is
  interpreted as a non-breaking-space and is preserved in the output (including
  always for UTF-8).

  The end result is that if the input is marked as an explicit endianness the
  transcoding is faithful to all codepoints in the source. If it is not marked
  with an explicit endianness, the BOM is not considered part of the string itself
  but as metadata, and so is not preserved in the output.

  Examples:

  >>> tf.strings.unicode_transcode(["Hello", "TensorFlow", "2.x"], "UTF-8", "UTF-16-BE")
  <tf.Tensor: shape=(3,), dtype=string, numpy=
  array([b'\\x00H\\x00e\\x00l\\x00l\\x00o',
         b'\\x00T\\x00e\\x00n\\x00s\\x00o\\x00r\\x00F\\x00l\\x00o\\x00w',
         b'\\x002\\x00.\\x00x'], dtype=object)>
  >>> tf.strings.unicode_transcode(["A", "B", "C"], "US ASCII", "UTF-8").numpy()
  array([b'A', b'B', b'C'], dtype=object)

  Args:
    input: A `Tensor` of type `string`.
      The text to be processed. Can have any shape.
    input_encoding: A `string`.
      Text encoding of the input strings. This is any of the encodings supported
      by ICU ucnv algorithmic converters. Examples: `"UTF-16", "US ASCII", "UTF-8"`.
    output_encoding: A `string` from: `"UTF-8", "UTF-16-BE", "UTF-32-BE"`.
      The unicode encoding to use in the output. Must be one of
      `"UTF-8", "UTF-16-BE", "UTF-32-BE"`. Multi-byte encodings will be big-endian.
    errors: An optional `string` from: `"strict", "replace", "ignore"`. Defaults to `"replace"`.
      Error handling policy when there is invalid formatting found in the input.
      The value of 'strict' will cause the operation to produce a InvalidArgument
      error on any invalid input formatting. A value of 'replace' (the default) will
      cause the operation to replace any invalid formatting in the input with the
      `replacement_char` codepoint. A value of 'ignore' will cause the operation to
      skip any invalid formatting in the input and produce no corresponding output
      character.
    replacement_char: An optional `int`. Defaults to `65533`.
      The replacement character codepoint to be used in place of any invalid
      formatting in the input when `errors='replace'`. Any valid unicode codepoint may
      be used. The default value is the default unicode replacement character is
      0xFFFD or U+65533.)

      Note that for UTF-8, passing a replacement character expressible in 1 byte, such
      as ' ', will preserve string alignment to the source since invalid bytes will be
      replaced with a 1-byte replacement. For UTF-16-BE and UTF-16-LE, any 1 or 2 byte
      replacement character will preserve byte alignment to the source.
    replace_control_characters: An optional `bool`. Defaults to `False`.
      Whether to replace the C0 control characters (00-1F) with the
      `replacement_char`. Default is false.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'UnicodeTranscode', name, input, 'input_encoding', input_encoding, 'output_encoding', output_encoding, 'errors', errors, 'replacement_char', replacement_char, 'replace_control_characters', replace_control_characters)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            _result = _dispatcher_for_unicode_transcode((input, input_encoding, output_encoding, errors, replacement_char, replace_control_characters, name), None)
            if _result is not NotImplemented:
                return _result
            return unicode_transcode_eager_fallback(input, input_encoding=input_encoding, output_encoding=output_encoding, errors=errors, replacement_char=replacement_char, replace_control_characters=replace_control_characters, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
        except (TypeError, ValueError):
            _result = _dispatch.dispatch(unicode_transcode, (), dict(input=input, input_encoding=input_encoding, output_encoding=output_encoding, errors=errors, replacement_char=replacement_char, replace_control_characters=replace_control_characters, name=name))
            if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
                return _result
            raise
    else:
        _result = _dispatcher_for_unicode_transcode((input, input_encoding, output_encoding, errors, replacement_char, replace_control_characters, name), None)
        if _result is not NotImplemented:
            return _result
    input_encoding = _execute.make_str(input_encoding, 'input_encoding')
    output_encoding = _execute.make_str(output_encoding, 'output_encoding')
    if errors is None:
        errors = 'replace'
    errors = _execute.make_str(errors, 'errors')
    if replacement_char is None:
        replacement_char = 65533
    replacement_char = _execute.make_int(replacement_char, 'replacement_char')
    if replace_control_characters is None:
        replace_control_characters = False
    replace_control_characters = _execute.make_bool(replace_control_characters, 'replace_control_characters')
    try:
        _, _, _op, _outputs = _op_def_library._apply_op_helper('UnicodeTranscode', input=input, input_encoding=input_encoding, output_encoding=output_encoding, errors=errors, replacement_char=replacement_char, replace_control_characters=replace_control_characters, name=name)
    except (TypeError, ValueError):
        _result = _dispatch.dispatch(unicode_transcode, (), dict(input=input, input_encoding=input_encoding, output_encoding=output_encoding, errors=errors, replacement_char=replacement_char, replace_control_characters=replace_control_characters, name=name))
        if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
            return _result
        raise
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('input_encoding', _op.get_attr('input_encoding'), 'output_encoding', _op.get_attr('output_encoding'), 'errors', _op.get_attr('errors'), 'replacement_char', _op._get_attr_int('replacement_char'), 'replace_control_characters', _op._get_attr_bool('replace_control_characters'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('UnicodeTranscode', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result