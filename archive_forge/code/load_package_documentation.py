import hashlib
import importlib.util
import os
import re
import subprocess
import tempfile
import yaml
import ray
Represents a Ray package loaded via ``load_package()``.

    This class provides access to the symbols defined by the interface file of
    the package (e.g., remote functions and actor definitions). You can also
    access the raw runtime env defined by the package via ``pkg._runtime_env``.
    