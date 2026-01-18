from __future__ import annotations
import json
import re
from traitlets.log import get_logger
from nbformat import v3, validator
from nbformat.corpus.words import generate_corpus_id as random_cell_id
from nbformat.notebooknode import NotebookNode
from .nbbase import nbformat, nbformat_minor
def upgrade_cell(cell):
    """upgrade a cell from v3 to v4

    heading cell:
        - -> markdown heading
    code cell:
        - remove language metadata
        - cell.input -> cell.source
        - cell.prompt_number -> cell.execution_count
        - update outputs
    """
    cell.setdefault('metadata', NotebookNode())
    cell.id = random_cell_id()
    if cell.cell_type == 'code':
        cell.pop('language', '')
        if 'collapsed' in cell:
            cell.metadata['collapsed'] = cell.pop('collapsed')
        cell.source = cell.pop('input', '')
        cell.execution_count = cell.pop('prompt_number', None)
        cell.outputs = upgrade_outputs(cell.outputs)
    elif cell.cell_type == 'heading':
        cell.cell_type = 'markdown'
        level = cell.pop('level', 1)
        cell.source = '{hashes} {single_line}'.format(hashes='#' * level, single_line=' '.join(cell.get('source', '').splitlines()))
    elif cell.cell_type == 'html':
        cell.cell_type = 'markdown'
    return cell