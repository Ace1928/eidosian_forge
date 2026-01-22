from __future__ import annotations
import logging  # isort:skip
from os import listdir
from os.path import join
from packaging.version import Version as V
from bokeh import __version__
from bokeh.resources import get_sri_hashes_for_version
from . import PARALLEL_SAFE
from .bokeh_directive import BokehDirective
from .templates import RELEASE_DETAIL
class BokehReleases(BokehDirective):

    def run(self):
        srcdir = self.env.app.srcdir
        versions = [x.rstrip('.rst') for x in listdir(join(srcdir, 'docs', 'releases')) if x.endswith('.rst')]
        versions.sort(key=V, reverse=True)
        rst = []
        for version in versions:
            try:
                hashes = get_sri_hashes_for_version(version)
                table = sorted(hashes.items())
            except ValueError:
                if version == __version__:
                    raise RuntimeError(f'Missing SRI Hash for full release version {version!r}')
                table = []
            rst.append(RELEASE_DETAIL.render(version=version, table=table))
        return self.parse('\n'.join(rst), '<bokeh-releases>')