import fnmatch
import json
import os
import shutil
import tempfile
import xml.etree.ElementTree as ET
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional
from datasets import config
from datasets.commands import BaseDatasetsCLICommand
from datasets.download.download_config import DownloadConfig
from datasets.download.download_manager import DownloadManager
from datasets.download.mock_download_manager import MockDownloadManager
from datasets.load import dataset_module_factory, import_main_class
from datasets.utils.deprecation_utils import deprecated
from datasets.utils.logging import get_logger, set_verbosity_warning
from datasets.utils.py_utils import map_nested
@staticmethod
def _create_xml_dummy_data(src_path, dst_path, xml_tag, n_lines=5, encoding=DEFAULT_ENCODING):
    Path(dst_path).parent.mkdir(exist_ok=True, parents=True)
    with open(src_path, encoding=encoding) as src_file:
        n_line = 0
        parents = []
        for event, elem in ET.iterparse(src_file, events=('start', 'end')):
            if event == 'start':
                parents.append(elem)
            else:
                _ = parents.pop()
                if elem.tag == xml_tag:
                    if n_line < n_lines:
                        n_line += 1
                    elif parents:
                        parents[-1].remove(elem)
        ET.ElementTree(element=elem).write(dst_path, encoding=encoding)