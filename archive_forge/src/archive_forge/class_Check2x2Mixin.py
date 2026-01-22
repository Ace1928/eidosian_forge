import os
import warnings
import numpy as np
import statsmodels.stats.contingency_tables as ctab
import pandas as pd
from numpy.testing import assert_allclose, assert_equal
import statsmodels.api as sm
class Check2x2Mixin:

    @classmethod
    def initialize(cls):
        cls.tbl_obj = ctab.Table2x2(cls.table)
        cls.tbl_data_obj = ctab.Table2x2.from_data(cls.data)

    def test_oddsratio(self):
        assert_allclose(self.tbl_obj.oddsratio, self.oddsratio)

    def test_log_oddsratio(self):
        assert_allclose(self.tbl_obj.log_oddsratio, self.log_oddsratio)

    def test_log_oddsratio_se(self):
        assert_allclose(self.tbl_obj.log_oddsratio_se, self.log_oddsratio_se)

    def test_oddsratio_pvalue(self):
        assert_allclose(self.tbl_obj.oddsratio_pvalue(), self.oddsratio_pvalue)

    def test_oddsratio_confint(self):
        lcb1, ucb1 = self.tbl_obj.oddsratio_confint(0.05)
        lcb2, ucb2 = self.oddsratio_confint
        assert_allclose(lcb1, lcb2)
        assert_allclose(ucb1, ucb2)

    def test_riskratio(self):
        assert_allclose(self.tbl_obj.riskratio, self.riskratio)

    def test_log_riskratio(self):
        assert_allclose(self.tbl_obj.log_riskratio, self.log_riskratio)

    def test_log_riskratio_se(self):
        assert_allclose(self.tbl_obj.log_riskratio_se, self.log_riskratio_se)

    def test_riskratio_pvalue(self):
        assert_allclose(self.tbl_obj.riskratio_pvalue(), self.riskratio_pvalue)

    def test_riskratio_confint(self):
        lcb1, ucb1 = self.tbl_obj.riskratio_confint(0.05)
        lcb2, ucb2 = self.riskratio_confint
        assert_allclose(lcb1, lcb2)
        assert_allclose(ucb1, ucb2)

    def test_log_riskratio_confint(self):
        lcb1, ucb1 = self.tbl_obj.log_riskratio_confint(0.05)
        lcb2, ucb2 = self.log_riskratio_confint
        assert_allclose(lcb1, lcb2)
        assert_allclose(ucb1, ucb2)

    def test_from_data(self):
        assert_equal(self.tbl_obj.summary().as_text(), self.tbl_data_obj.summary().as_text())

    def test_summary(self):
        assert_equal(self.tbl_obj.summary().as_text(), self.summary_string)