from __future__ import annotations
import logging
import re
from argparse import ArgumentParser
from pathlib import Path
from shutil import move, rmtree
from subprocess import check_call
from tempfile import TemporaryDirectory
from textwrap import dedent
def apply_code_modifications(source_path: Path):
    move(source_path / '__version__.py', source_path / '_version.py')
    for file in source_path.rglob('*.py'):
        content = file.read_text()
        content_orig = content
        content = re.sub('from .__version__ import VERSION', 'from ._version import VERSION', content)
        content = re.sub('import requests', dedent('\n                try:\n                    import requests\n                except ImportError:\n                    requests = None\n                ').strip(), content)
        content = re.sub('from dateutil.parser import isoparse', dedent('\n                try:\n                    from dateutil.parser import isoparse\n                except ImportError:\n                    isoparse = None\n                ').strip(), content)
        content = re.sub(': requests\\.Response', '', content)
        if content != content_orig:
            logger.info('Applied code modifications on %s', file)
        file.write_text(content)