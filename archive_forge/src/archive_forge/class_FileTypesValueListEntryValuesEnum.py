from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FileTypesValueListEntryValuesEnum(_messages.Enum):
    """FileTypesValueListEntryValuesEnum enum type.

    Values:
      FILE_TYPE_UNSPECIFIED: Includes all files.
      BINARY_FILE: Includes all file extensions not covered by another entry.
        Binary scanning attempts to convert the content of the file to utf_8
        to scan the file. If you wish to avoid this fall back, specify one or
        more of the other file types in your storage scan.
      TEXT_FILE: Included file extensions: asc,asp, aspx, brf, c, cc,cfm, cgi,
        cpp, csv, cxx, c++, cs, css, dart, dat, dot, eml,, epbub, ged, go, h,
        hh, hpp, hxx, h++, hs, html, htm, mkd, markdown, m, ml, mli, perl, pl,
        plist, pm, php, phtml, pht, properties, py, pyw, rb, rbw, rs, rss, rc,
        scala, sh, sql, swift, tex, shtml, shtm, xhtml, lhs, ics, ini, java,
        js, json, jsonl, kix, kml, ocaml, md, txt, text, tsv, vb, vcard, vcs,
        wml, xcodeproj, xml, xsl, xsd, yml, yaml.
      IMAGE: Included file extensions: bmp, gif, jpg, jpeg, jpe, png. Setting
        bytes_limit_per_file or bytes_limit_per_file_percent has no effect on
        image files. Image inspection is restricted to the `global`, `us`,
        `asia`, and `europe` regions.
      WORD: Microsoft Word files larger than 30 MB will be scanned as binary
        files. Included file extensions: docx, dotx, docm, dotm. Setting
        `bytes_limit_per_file` or `bytes_limit_per_file_percent` has no effect
        on Word files.
      PDF: PDF files larger than 30 MB will be scanned as binary files.
        Included file extensions: pdf. Setting `bytes_limit_per_file` or
        `bytes_limit_per_file_percent` has no effect on PDF files.
      AVRO: Included file extensions: avro
      CSV: Included file extensions: csv
      TSV: Included file extensions: tsv
      POWERPOINT: Microsoft PowerPoint files larger than 30 MB will be scanned
        as binary files. Included file extensions: pptx, pptm, potx, potm,
        pot. Setting `bytes_limit_per_file` or `bytes_limit_per_file_percent`
        has no effect on PowerPoint files.
      EXCEL: Microsoft Excel files larger than 30 MB will be scanned as binary
        files. Included file extensions: xlsx, xlsm, xltx, xltm. Setting
        `bytes_limit_per_file` or `bytes_limit_per_file_percent` has no effect
        on Excel files.
    """
    FILE_TYPE_UNSPECIFIED = 0
    BINARY_FILE = 1
    TEXT_FILE = 2
    IMAGE = 3
    WORD = 4
    PDF = 5
    AVRO = 6
    CSV = 7
    TSV = 8
    POWERPOINT = 9
    EXCEL = 10