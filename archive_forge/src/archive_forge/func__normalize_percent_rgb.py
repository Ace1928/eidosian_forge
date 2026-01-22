from . import constants, types
def _normalize_percent_rgb(value: str) -> str:
    """
    Internal normalization function for clipping percent values into
    the permitted range (0%-100%, inclusive).

    """
    value = value.split('%')[0]
    percent = float(value) if '.' in value else int(value)
    return '0%' if percent < 0 else '100%' if percent > 100 else f'{percent}%'