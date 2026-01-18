import gzip
import io
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import textwrap
import time
import zipfile
import numpy as np
import pytest
from pandas.compat import is_platform_windows
import pandas as pd
import pandas._testing as tm
import pandas.io.common as icom
@pytest.mark.single_cpu
def test_with_missing_lzma_runtime():
    """Tests if RuntimeError is hit when calling lzma without
    having the module available.
    """
    code = textwrap.dedent("\n        import sys\n        import pytest\n        sys.modules['lzma'] = None\n        import pandas as pd\n        df = pd.DataFrame()\n        with pytest.raises(RuntimeError, match='lzma module'):\n            df.to_csv('foo.csv', compression='xz')\n        ")
    subprocess.check_output([sys.executable, '-c', code], stderr=subprocess.PIPE)