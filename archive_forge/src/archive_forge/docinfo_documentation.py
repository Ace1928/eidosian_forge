from __future__ import annotations
from without sublassing the scanner.

    Store document information, can be used for analysis of a loaded YAML document
    requested_version: if explicitly set before load
    doc_version: from %YAML directive
    tags: from %TAG directives in scanned order
    