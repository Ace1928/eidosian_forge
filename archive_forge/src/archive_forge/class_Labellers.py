import pytest
from ...labels import (
class Labellers:

    def __init__(self):
        self.labellers = {'BaseLabeller': BaseLabeller(), 'DimCoordLabeller': DimCoordLabeller(), 'IdxLabeller': IdxLabeller(), 'DimIdxLabeller': DimIdxLabeller(), 'MapLabeller': MapLabeller(), 'NoVarLabeller': NoVarLabeller(), 'NoModelLabeller': NoModelLabeller()}