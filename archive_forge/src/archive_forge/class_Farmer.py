import os
import sys
import time
from pyomo.common.dependencies import mpi4py
from pyomo.contrib.benders.benders_cuts import BendersCutGenerator
import pyomo.environ as pyo
class Farmer(object):

    def __init__(self):
        self.crops = ['WHEAT', 'CORN', 'SUGAR_BEETS']
        self.total_acreage = 500
        self.PriceQuota = {'WHEAT': 100000.0, 'CORN': 100000.0, 'SUGAR_BEETS': 6000.0}
        self.SubQuotaSellingPrice = {'WHEAT': 170.0, 'CORN': 150.0, 'SUGAR_BEETS': 36.0}
        self.SuperQuotaSellingPrice = {'WHEAT': 0.0, 'CORN': 0.0, 'SUGAR_BEETS': 10.0}
        self.CattleFeedRequirement = {'WHEAT': 200.0, 'CORN': 240.0, 'SUGAR_BEETS': 0.0}
        self.PurchasePrice = {'WHEAT': 238.0, 'CORN': 210.0, 'SUGAR_BEETS': 100000.0}
        self.PlantingCostPerAcre = {'WHEAT': 150.0, 'CORN': 230.0, 'SUGAR_BEETS': 260.0}
        self.scenarios = ['BelowAverageScenario', 'AverageScenario', 'AboveAverageScenario']
        self.crop_yield = dict()
        self.crop_yield['BelowAverageScenario'] = {'WHEAT': 2.0, 'CORN': 2.4, 'SUGAR_BEETS': 16.0}
        self.crop_yield['AverageScenario'] = {'WHEAT': 2.5, 'CORN': 3.0, 'SUGAR_BEETS': 20.0}
        self.crop_yield['AboveAverageScenario'] = {'WHEAT': 3.0, 'CORN': 3.6, 'SUGAR_BEETS': 24.0}
        self.scenario_probabilities = dict()
        self.scenario_probabilities['BelowAverageScenario'] = 0.3333
        self.scenario_probabilities['AverageScenario'] = 0.3334
        self.scenario_probabilities['AboveAverageScenario'] = 0.3333