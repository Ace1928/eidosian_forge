import os
from pathlib import Path
import re
import subprocess
import sys
import matplotlib.pyplot as plt
from matplotlib.texmanager import TexManager
from matplotlib.testing._markers import needs_usetex
import pytest
@needs_usetex
def test_openin_any_paranoid():
    completed = subprocess.run([sys.executable, '-c', 'import matplotlib.pyplot as plt;plt.rcParams.update({"text.usetex": True});plt.title("paranoid");plt.show(block=False);'], env={**os.environ, 'openin_any': 'p'}, check=True, capture_output=True)
    assert completed.stderr == b''