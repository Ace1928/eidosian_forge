from __future__ import annotations
import hashlib
import io
import os
import re
from pathlib import Path
from streamlit import runtime
from streamlit.runtime import caching
Handles io.BytesIO data, converting SRT to VTT content if needed.