from __future__ import annotations
import logging
import re
from argparse import ArgumentParser
from pathlib import Path
from shutil import move, rmtree
from subprocess import check_call
from tempfile import TemporaryDirectory
from textwrap import dedent

Fetch and bundles the hcloud package inside the collection.

Fetch the desired version `HCLOUD_VERSION` from https://github.com/hetznercloud/hcloud-python
`HCLOUD_SOURCE_URL` using git, apply some code modifications to comply with ansible,
move the modified files at the vendor location `HCLOUD_VENDOR_PATH`.
