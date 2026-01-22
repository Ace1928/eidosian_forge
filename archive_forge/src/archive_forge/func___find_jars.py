import sys
import logging
import re
import tempfile
import xml.etree.ElementTree as ET
import zipfile
import PySide6
from pathlib import Path
from typing import List
from pkginfo import Wheel
from .. import MAJOR_VERSION, BaseConfig, Config, run_command
from . import (create_recipe, find_lib_dependencies, find_qtlibs_in_wheel,
def __find_jars(self, dependency_files: List[zipfile.Path], jars_dir: Path):
    jars, init_classes = (set(), set())
    for dependency_file in dependency_files:
        xml_content = dependency_file.read_text()
        root = ET.fromstring(xml_content)
        for jar in root.iter('jar'):
            jar_file = jar.attrib['file']
            if jar_file.startswith('jar/'):
                jar_file_name = jar_file[4:]
                if (jars_dir / jar_file_name).exists():
                    jars.add(str(jars_dir / jar_file_name))
                else:
                    logging.warning(f'[DEPLOY] Unable to include {jar_file}. {jar_file} does not exist in {jars_dir}')
                    continue
            else:
                logging.warning(f"[DEPLOY] Unable to include {jar_file}. All jar file paths should begin with 'jar/'")
                continue
            jar_init_class = jar.attrib.get('initClass')
            if jar_init_class:
                init_classes.add(jar_init_class)
    android_bindings_jar = jars_dir / 'Qt6AndroidBindings.jar'
    if android_bindings_jar.exists():
        jars.add(str(android_bindings_jar))
    else:
        raise FileNotFoundError(f'{android_bindings_jar} not found in wheel')
    return (jars, init_classes)