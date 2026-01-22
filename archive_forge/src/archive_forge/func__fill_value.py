import packaging.version
import pandas as pd
def _fill_value(self):
    try:
        return self.values.dtype.fill_value
    except AttributeError:
        return self.values.dtype.na_value