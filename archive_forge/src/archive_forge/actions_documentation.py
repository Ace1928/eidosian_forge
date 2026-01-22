from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import io
import os
import sys
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.core import log
from googlecloudsdk.core import metrics
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
import six
Render a help document according to the style in values.

      Args:
        parser: The ArgParse object.
        namespace: The ArgParse namespace.
        values: The --document flag ArgDict() value:
          style=STYLE
            The output style. Must be specified.
          title=DOCUMENT TITLE
            The document title.
          notes=SENTENCES
            Inserts SENTENCES into the document NOTES section.
        option_string: The ArgParse flag string.

      Raises:
        parser_errors.ArgumentError: For unknown flag value attribute name.
      