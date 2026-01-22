from .specs import SPEC_BY_TYPE, make_msgdict
def _parse_time(value):
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    raise ValueError(f'invalid time {value!r}')