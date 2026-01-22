import configparser
import contextlib
import locale
import logging
import pathlib
import re
import sys
from itertools import chain, groupby, repeat
from typing import TYPE_CHECKING, Dict, Iterator, List, Optional, Union
from pip._vendor.requests.models import Request, Response
from pip._vendor.rich.console import Console, ConsoleOptions, RenderResult
from pip._vendor.rich.markup import escape
from pip._vendor.rich.text import Text
class MetadataGenerationFailed(InstallationSubprocessError, InstallationError):
    reference = 'metadata-generation-failed'

    def __init__(self, *, package_details: str) -> None:
        super(InstallationSubprocessError, self).__init__(message='Encountered error while generating package metadata.', context=escape(package_details), hint_stmt='See above for details.', note_stmt='This is an issue with the package mentioned above, not pip.')

    def __str__(self) -> str:
        return 'metadata generation failed'