#
# diffoscope: in-depth comparison of files, archives, and directories
#
# Copyright © 2022 Christopher Baines <mail@cbaines.net>
# Copyright © 2022 Chris Lamb <lamby@debian.org>
#
# diffoscope is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# diffoscope is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with diffoscope.  If not, see <https://www.gnu.org/licenses/>.

import re
import os.path
import logging
import subprocess

from diffoscope.tools import tool_required

from .utils.file import File
from .utils.archive import Archive

logger = logging.getLogger(__name__)


class LzipContainer(Archive):
    def open_archive(self):
        return self

    def close_archive(self):
        pass

    def get_member_names(self):
        return [self.get_compressed_content_name(".lzip")]

    @tool_required("lzip")
    def extract(self, member_name, dest_dir):
        dest_path = os.path.join(dest_dir, member_name)
        logger.debug("lzip extracting to %s", dest_path)
        with open(dest_path, "wb") as fp:
            subprocess.check_call(
                ["lzip", "--decompress", "--stdout", self.source.path],
                stdout=fp,
                stderr=None,
            )
        return dest_path


class LzipFile(File):
    DESCRIPTION = "lzip compressed files"
    CONTAINER_CLASSES = [LzipContainer]
    FILE_TYPE_RE = re.compile(r"^lzip compressed data\b")

    FALLBACK_FILE_EXTENSION_SUFFIX = {".lzip"}
