import ast
import collections
import os
import re
import shutil
import sys
import tempfile
import traceback
import pasta
def _get_applicable_dict(self, transformer_field, full_name, name):
    """Get all dict entries indexed by name that apply to full_name or name."""
    function_transformers = getattr(self._api_change_spec, transformer_field, {})
    glob_name = '*.' + name if name else None
    transformers = function_transformers.get('*', {}).copy()
    transformers.update(function_transformers.get(glob_name, {}))
    transformers.update(function_transformers.get(full_name, {}))
    return transformers