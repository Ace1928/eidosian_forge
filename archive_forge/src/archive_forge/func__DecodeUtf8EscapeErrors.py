import re
def _DecodeUtf8EscapeErrors(text_bytes):
    ret = ''
    while text_bytes:
        try:
            ret += text_bytes.decode('utf-8').translate(_str_escapes)
            text_bytes = ''
        except UnicodeDecodeError as e:
            ret += text_bytes[:e.start].decode('utf-8').translate(_str_escapes)
            ret += _byte_escapes[text_bytes[e.start]]
            text_bytes = text_bytes[e.start + 1:]
    return ret