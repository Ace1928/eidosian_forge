#
# diffoscope: in-depth comparison of files, archives, and directories
#
# Copyright © 2024 Seth Michael Larson <seth@python.org>
# Copyright © 2024 Chris Lamb <lamby@debian.org>
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
import hashlib
import re
import logging
import struct
import xml.etree.ElementTree as ET
import zlib
import os

from .utils.file import File
from .utils.archive import Archive
from diffoscope.difference import Difference

logger = logging.getLogger(__name__)


class XarContainer(Archive):
    def get_member_names(self):
        toc_xml = self.parse_toc_xml()
        for file_tag in toc_xml.iter("file"):
            yield file_tag.get("id")  # Use IDs instead of names for members.

    def open_archive(self):
        pass

    def close_archive(self):
        pass

    def extract(self, member_name, dest_dir):
        toc_xml = self.parse_toc_xml()

        # Find the file name and data heap offset.
        for file_tag in toc_xml.iter("file"):
            if file_tag.get("id") == member_name:
                file_name = file_tag.find(".//name").text
                data_offset = int(file_tag.find(".//data/offset").text)
                data_length = int(file_tag.find(".//data/length").text)
                break
        else:
            raise KeyError(member_name)

        # Write data from the heap into the temporary directory.
        # We automatically handle gzipped data thanks to the header.
        dest_path = os.path.join(dest_dir, file_name)
        with open(dest_path, mode="wb") as fw, open(
            self._source.path, mode="rb"
        ) as fr:
            fr.seek(self._heap_offset + data_offset, 0)
            fw.write(fr.read(data_length))

        return dest_path

    def parse_toc_xml(self):
        if getattr(self, "_toc_xml", None) is None:
            with open(self._source.path, mode="rb") as f:
                # Skip the magic and format, we're looking for
                # header and TOC compressed lengths.
                (
                    header_length,
                    toc_compressed_length,
                ) = struct.unpack(">xxxxHxxQ", f.read(16))

                # Read, decompress, and parse the TOC as XML. Save heap offset for later.
                f.seek(header_length, 0)
                toc_bytes = f.read(toc_compressed_length)
                toc_as_text = zlib.decompress(toc_bytes).decode("utf-8")
                self._toc_xml = ET.XML(toc_as_text)
                self._heap_offset = header_length + toc_compressed_length

        return self._toc_xml


class XarFile(File):
    DESCRIPTION = "eXtensible ARchive files"
    CONTAINER_CLASSES = [XarContainer]
    FILE_TYPE_RE = re.compile(r"\bpkg\b")
    FALLBACK_FILE_EXTENSION_SUFFIX = {
        ".xar",
        ".pkg",
    }  # NOTE: Facebook's Executable Archive format also uses '.xar'.
    FALLBACK_FILE_TYPE_HEADER_PREFIX = b"xar!"

    def compare_details(self, other, source=None):
        self_xar_header, self_xar_toc = describe_xar(self.path)
        other_xar_header, other_xar_toc = describe_xar(other.path)
        return [
            Difference.from_text(
                self_xar_header,
                other_xar_header,
                self.path,
                other.path,
                source="XAR Header",
            ),
            Difference.from_text(
                self_xar_toc,
                other_xar_toc,
                self.path,
                other.path,
                source="XAR Table of Contents",
            ),
        ]


def describe_xar(path):
    with open(path, mode="rb") as f:
        magic = f.read(4)

        # Read the fixed portion of the XAR header
        # Padding length is calculated using header length.
        (
            header_length,
            format_version,
            toc_compressed_length,
            toc_uncompressed_length,
            checksum_alg,
        ) = struct.unpack(">HHQQI", f.read(24))

        known_checksum_algs = {
            0: "NONE",
            1: "SHA1",
            2: "MD5",
            3: "SHA-256",
            4: "SHA-512",
        }

        header_lines = [
            "magic:                   {}".format(magic),
            "format version:          {}".format(format_version),
            "TOC compressed length:   {}".format(toc_compressed_length),
            "TOC uncompressed length: {}".format(toc_uncompressed_length),
            "checksum:                {} ({})".format(
                checksum_alg, known_checksum_algs.get(checksum_alg, "???")
            ),
        ]

        # Note that this 'header length' includes the 4 bytes of magic, hence 28.
        padding_length = header_length - 28
        if padding_length > 0:  # Padding is optional.
            padding = f.read(padding_length)
            header_lines.append("padding:                 {}".format(padding))

        # Read the TOC which is always DEFLATE compressed.
        toc_bytes = f.read(toc_compressed_length)
        toc_as_text = zlib.decompress(toc_bytes).decode("utf-8")

        # Read the entire heap and add properties that allow detecting
        # "invisible" differences in the heap, for example if data is inserted
        # but isn't referenced in the TOC. This shouldn't happen in a normal XAR file.
        heap_bytes = f.read()
        header_lines.extend(
            [
                "heap length:             {}".format(len(heap_bytes)),
                "heap checksum:           {}".format(
                    hashlib.sha256(heap_bytes).hexdigest()
                ),
            ]
        )

        header_as_text = "\n".join(header_lines)
        return header_as_text, toc_as_text
