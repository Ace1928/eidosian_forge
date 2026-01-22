import inspect
import re
import urllib
from typing import List as LList
from typing import Optional, Union
from .... import __version__ as wandb_ver
from .... import termwarn
from ...public import Api as PublicApi
from ._panels import UnknownPanel, WeavePanel, panel_mapping, weave_panels
from .runset import Runset
from .util import (
from .validators import OneOf, TypeValidator
@staticmethod
def _default_panel_grid_spec():
    return {'type': 'panel-grid', 'children': [{'text': ''}], 'metadata': {'openViz': True, 'panels': {'views': {'0': {'name': 'Panels', 'defaults': [], 'config': []}}, 'tabs': ['0']}, 'panelBankConfig': {'state': 0, 'settings': {'autoOrganizePrefix': 2, 'showEmptySections': False, 'sortAlphabetically': False}, 'sections': [{'name': 'Hidden Panels', 'isOpen': False, 'panels': [], 'type': 'flow', 'flowConfig': {'snapToColumns': True, 'columnsPerPage': 3, 'rowsPerPage': 2, 'gutterWidth': 16, 'boxWidth': 460, 'boxHeight': 300}, 'sorted': 0, 'localPanelSettings': {'xAxis': '_step', 'smoothingWeight': 0, 'smoothingType': 'exponential', 'ignoreOutliers': False, 'xAxisActive': False, 'smoothingActive': False}}]}, 'panelBankSectionConfig': {'name': 'Report Panels', 'isOpen': False, 'panels': [], 'type': 'grid', 'flowConfig': {'snapToColumns': True, 'columnsPerPage': 3, 'rowsPerPage': 2, 'gutterWidth': 16, 'boxWidth': 460, 'boxHeight': 300}, 'sorted': 0, 'localPanelSettings': {'xAxis': '_step', 'smoothingWeight': 0, 'smoothingType': 'exponential', 'ignoreOutliers': False, 'xAxisActive': False, 'smoothingActive': False}}, 'customRunColors': {}, 'runSets': [], 'openRunSet': 0, 'name': 'unused-name'}}