from pathlib import Path
import urllib.parse
import os
from typing import Union
Returns a new URI that strips the given subpath from the end of this URI.

        Example:
            >>> uri = URI("s3://bucket/a/b/c/?param=1")
            >>> str(uri.rstrip_subpath(Path("b/c")))
            's3://bucket/a?param=1'

            >>> uri = URI("/tmp/a/b/c/")
            >>> str(uri.rstrip_subpath(Path("/b/c/.//")))
            '/tmp/a'

        