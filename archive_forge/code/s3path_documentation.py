import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING
from ..cloudpath import CloudPath, NoStatError, register_path_class
Class for representing and operating on AWS S3 URIs, in the style of the Python standard
    library's [`pathlib` module](https://docs.python.org/3/library/pathlib.html). Instances
    represent a path in S3 with filesystem path semantics, and convenient methods allow for basic
    operations like joining, reading, writing, iterating over contents, etc. This class almost
    entirely mimics the [`pathlib.Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)
    interface, so most familiar properties and methods should be available and behave in the
    expected way.

    The [`S3Client`](../s3client/) class handles authentication with AWS. If a client instance is
    not explicitly specified on `S3Path` instantiation, a default client is used. See `S3Client`'s
    documentation for more details.
    