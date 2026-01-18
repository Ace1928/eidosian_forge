from __future__ import annotations
import asyncio
import glob
import logging
import os
import sys
import typing as t
from textwrap import dedent, fill
from jupyter_core.application import JupyterApp, base_aliases, base_flags
from traitlets import Bool, DottedObjectName, Instance, List, Type, Unicode, default, observe
from traitlets.config import Configurable, catch_config_error
from traitlets.utils.importstring import import_item
from nbconvert import __version__, exporters, postprocessors, preprocessors, writers
from nbconvert.utils.text import indent
from .exporters.base import get_export_names, get_exporter
from .utils.base import NbConvertBase
from .utils.exceptions import ConversionException
from .utils.io import unicode_stdin_stream
def write_single_notebook(self, output, resources):
    """Step 3: Write the notebook to file

        This writes output from the exporter to file using the specified writer.
        It returns the results from the writer.

        Parameters
        ----------
        output :
        resources : dict
            resources for a single notebook including name, config directory
            and directory to save output

        Returns
        -------
        file
            results from the specified writer output of exporter
        """
    if 'unique_key' not in resources:
        msg = 'unique_key MUST be specified in the resources, but it is not'
        raise KeyError(msg)
    notebook_name = resources['unique_key']
    if self.use_output_suffix and self.output_base == '{notebook_name}':
        notebook_name += resources.get('output_suffix', '')
    if not self.writer:
        msg = 'No writer object defined!'
        raise ValueError(msg)
    return self.writer.write(output, resources, notebook_name=notebook_name)