from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
import sys
from googlecloudsdk.calliope import base
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.document_renderers import render_document
Uses gcloud's markdown renderer to render the given markdown file.