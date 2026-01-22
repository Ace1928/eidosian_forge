import base64
import inspect
import builtins
@classmethod
def _get_default_encoder(cls, encode_string):

    def _encode(v):
        if isinstance(v, (bytes, str)):
            if isinstance(v, str):
                v = v.encode('utf-8')
            json_value = encode_string(v)
            json_value = json_value.decode('ascii')
        elif isinstance(v, list):
            json_value = [_encode(ve) for ve in v]
        elif isinstance(v, dict):
            json_value = _mapdict(_encode, v)
            json_value = _mapdict_key(str, json_value)
            assert not cls._is_class(json_value)
        else:
            try:
                json_value = v.to_jsondict()
            except Exception:
                json_value = v
        return json_value
    return _encode