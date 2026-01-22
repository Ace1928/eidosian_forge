import os
import logging
import pyomo.contrib.viewer.qt as myqt
from pyomo.contrib.viewer.report import value_no_exception, get_residual
import pyomo.environ as pyo
from pyomo.common.fileutils import this_file_dir
class ResidualTable(_ResidualTable, _ResidualTableUI):

    def __init__(self, ui_data):
        super().__init__()
        self.setupUi(self)
        self.ui_data = ui_data
        datmodel = ResidualDataModel(parent=self, ui_data=ui_data)
        self.datmodel = datmodel
        self.tableView.setModel(datmodel)
        self.ui_data.updated.connect(self.refresh)
        self.sortButton.clicked.connect(self.sort)
        self.calculateButton.clicked.connect(self.calculate)

    def sort(self):
        self.datmodel.sort()

    def refresh(self):
        self.datmodel.update_model()
        self.datmodel.sort()
        self.datmodel.layoutChanged.emit()

    def calculate(self):
        self.ui_data.calculate_constraints()
        self.refresh()