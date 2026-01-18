import logging
from pathlib import Path
from types import SimpleNamespace
from typing import List
import pytest
from jupyter_lsp import LanguageServerManager
from ..virtual_documents_shadow import (
We call server well behaved when it does not require a disk copy