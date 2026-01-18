import contextlib
from pathlib import Path
from unittest import mock
import pytest
import cartopy
import cartopy.io as cio
from cartopy.io.shapereader import NEShpDownloader

    Context manager which defaults the "data_dir" to a temporary directory
    which is automatically cleaned up on exit.

    