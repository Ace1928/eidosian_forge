import numpy as np
def _datetime_metadata_str(dtype):
    unit, count = np.datetime_data(dtype)
    if unit == 'generic':
        return ''
    elif count == 1:
        return '[{}]'.format(unit)
    else:
        return '[{}{}]'.format(count, unit)