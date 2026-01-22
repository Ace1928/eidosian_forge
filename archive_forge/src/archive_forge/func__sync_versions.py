from __future__ import absolute_import, division, print_function
import argparse
import gzip
import pathlib
import shutil
import subprocess
import sys
from urllib import request
from xml.etree import ElementTree
import yaml
def _sync_versions(vars, available_versions, cache_dir):
    new_vars = dict(vars, _msi_lookup={})
    old_msis = vars['_msi_lookup']
    new_msis = new_vars['_msi_lookup']
    cache = pathlib.Path(cache_dir)
    for version in sorted(available_versions):
        version_str = '.'.join(map(str, version[:3]))
        if version_str in old_msis:
            new_msis[version_str] = old_msis[version_str]
            continue
        product_codes = {}
        for arch in ('x86', 'x64'):
            url = DOWNLOAD_URL_TEMPLATE.format(*version, arch=arch)
            filename = FILENAME_TEMPLATE.format(*version, arch=arch)
            file = cache / filename
            if not file.is_file():
                print('Downloading ' + filename)
                with open(file, 'wb') as fp:
                    response = request.urlopen(url)
                    shutil.copyfileobj(response, fp)
            else:
                print('Reusing ' + filename)
            process = subprocess.run(('msiinfo', 'export', str(file), 'Property'), capture_output=True, check=True)
            for line in process.stdout.splitlines():
                field, value = line.split(b'\t')
                if field == b'ProductCode':
                    product_codes[arch] = value.decode('ascii')
        new_msis[version_str] = dict(version=version_str, build=version[-1], product_codes=product_codes)
    new_msis['latest'] = new_msis[version_str]
    return new_vars