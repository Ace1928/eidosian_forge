from .._utils import set_module
@staticmethod
def _size_to_string(num_bytes):
    """ Convert a number of bytes into a binary size string """
    LOG2_STEP = 10
    STEP = 1024
    units = ['bytes', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB', 'EiB']
    unit_i = max(num_bytes.bit_length() - 1, 1) // LOG2_STEP
    unit_val = 1 << unit_i * LOG2_STEP
    n_units = num_bytes / unit_val
    del unit_val
    if round(n_units) == STEP:
        unit_i += 1
        n_units /= STEP
    if unit_i >= len(units):
        new_unit_i = len(units) - 1
        n_units *= 1 << (unit_i - new_unit_i) * LOG2_STEP
        unit_i = new_unit_i
    unit_name = units[unit_i]
    if unit_i == 0:
        return '{:.0f} {}'.format(n_units, unit_name)
    elif round(n_units) < 1000:
        return '{:#.3g} {}'.format(n_units, unit_name)
    else:
        return '{:#.0f} {}'.format(n_units, unit_name)