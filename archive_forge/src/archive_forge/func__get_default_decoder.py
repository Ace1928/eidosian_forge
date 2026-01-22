import base64
import inspect
import builtins
@classmethod
def _get_default_decoder(cls, decode_string):

    def _decode(json_value, **additional_args):
        if isinstance(json_value, (bytes, str)):
            v = decode_string(json_value)
        elif isinstance(json_value, list):
            v = [_decode(jv) for jv in json_value]
        elif isinstance(json_value, dict):
            if cls._is_class(json_value):
                v = cls.obj_from_jsondict(json_value, **additional_args)
            else:
                v = _mapdict(_decode, json_value)
                try:
                    v = _mapdict_key(int, v)
                except ValueError:
                    pass
        else:
            v = json_value
        return v
    return _decode