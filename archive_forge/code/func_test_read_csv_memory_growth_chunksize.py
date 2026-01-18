from io import StringIO
import numpy as np
import pytest
from pandas._libs import parsers as libparsers
from pandas.errors import DtypeWarning
from pandas import (
import pandas._testing as tm
def test_read_csv_memory_growth_chunksize(all_parsers):
    parser = all_parsers
    with tm.ensure_clean() as path:
        with open(path, 'w', encoding='utf-8') as f:
            for i in range(1000):
                f.write(str(i) + '\n')
        if parser.engine == 'pyarrow':
            msg = "The 'chunksize' option is not supported with the 'pyarrow' engine"
            with pytest.raises(ValueError, match=msg):
                with parser.read_csv(path, chunksize=20) as result:
                    for _ in result:
                        pass
            return
        with parser.read_csv(path, chunksize=20) as result:
            for _ in result:
                pass